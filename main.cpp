//#define BENCHMARK
#include "FM32Kernel.cuh"

#ifndef BENCHMARK

#define IMGUI_IMPL_OPENGL_LOADER_GLEW
#define __WINDOWS_MM__

#include "InitStuffs.h"

#include "Libs/imgui/imgui.h"
#include "Libs/imgui/examples/imgui_impl_sdl.h"
#include "Libs/imgui/examples/imgui_impl_opengl3.h"

#include "Libs/rtmidi/RtMidi.h"

#include "SDL.h"
#include "GL/glew.h"
#endif

#include <chrono>
#include <fstream>
#include <memory>
#include <random>


int main(int argc, char** args)
{
#ifndef BENCHMARK
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
    {
        printf("Error: %s\n", SDL_GetError());
        return -1;
    }

    std::unique_ptr<RtMidiIn> midiIn;


    auto initParams = FM32::InitSDL();
    SDL_Window* window = std::get<0>(initParams);
    SDL_GLContext gl_context = std::get<1>(initParams);
    const char* glsl_version = std::get<2>(initParams);
    //auto [window, gl_context, glsl_version] = FM32::InitSDL(); // C++17, not yet supported by NVCC

    bool err = glewInit() != GLEW_OK;
    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();// (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImVec4 clear_color = ImVec4(0.25f, 0.35f, 0.40f, 1.00f);

#endif

    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Failed to init GPU!\n");
		return 1;
	}

    TestReduceFM();

    constexpr size_t sampleRate = 48000;
    constexpr size_t samplesPerFrame = 256;
    constexpr size_t numChannels = 1;
    size_t noteSlots[128];

#ifndef BENCHMARK
    FMSynth fm32(sampleRate, samplesPerFrame, numChannels);

    // Debugging last chance
    //fm32.allInUse();


    SDL_AudioSpec desiredSpec, actualSpec;
    desiredSpec.callback = [](void* userData, Uint8* buffer, int nSamples)
    {
        static bool overrun = false;
        if (overrun)
        {
            printf("Audio overrun!!");
        }
        FMSynth& fm32 = *(FMSynth *)userData;
        std::lock_guard<std::mutex> synthLock(fm32.parameterMutex);
        overrun = true;

        float* audioBuffer = (float*)buffer;
        
        fm32.getRenderedAudio(audioBuffer);
        fm32.renderAudio(); // Audio rendering will happen asynchronously due to CUDA
        overrun = false;
    };

    desiredSpec.userdata = &fm32;
    desiredSpec.channels = 1;
    desiredSpec.freq = sampleRate;
    desiredSpec.samples = samplesPerFrame;
    desiredSpec.format = AUDIO_F32SYS;

    if (SDL_OpenAudio(&desiredSpec, &actualSpec) < 0) {
        fprintf(stderr, "Couldn't open audio: %s\n", SDL_GetError());
        exit(-1);
    }

    assert(desiredSpec.format == actualSpec.format);
    fm32.setSampleRate(actualSpec.freq);
    fm32.setNumChannels(actualSpec.channels);
    fm32.setSamplesPerFrame(actualSpec.samples);

    std::future<void> fileFuture;
    
    SDL_PauseAudio(false);
    
    bool done = false;
    while (!done)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                done = true;
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window))
                done = true;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();

        if (ImGui::Begin("File and MIDI I/O"))
        {
            ImGui::Text("Patch Files");
            if (ImGui::Button("Load Patch..."))
            {
                if (fileFuture.valid()) {
                    fileFuture.wait();
                }
                fileFuture = std::async(std::launch::async, [&fm32]()
                {
                    // TODO: Properly Invoke OS load panel

                    std::ifstream loadFile("Patch.fm32");
                    std::lock_guard<std::mutex> paramLock(fm32.parameterMutex);
                    loadFile.read((char *)&fm32.parameterEnvelopes, sizeof(SynthParameterPack));
                });
            }
            
            if (ImGui::Button("Save Patch..."))
            {
                if (fileFuture.valid()) {
                    fileFuture.wait();
                }
                fileFuture = std::async(std::launch::async, [params = fm32.parameterEnvelopes]()
                {
                    // TODO: Properly Invoke OS save panel
                    
                    std::ofstream saveFile("Patch.fm32");
                    saveFile.write((const char*)&params, sizeof(SynthParameterPack));
                });
            }

            if (ImGui::Button("Clear Patch"))
            {
                std::lock_guard<std::mutex> paramLock(fm32.parameterMutex);
                fm32.parameterEnvelopes.freqMode.reset();
                fm32.parameterEnvelopes.modulationMatrix.clear();
                fm32.parameterEnvelopes.operatorVolume.clear();
                fm32.parameterEnvelopes.perOperatorFreq.clear();
            }

            if (ImGui::Button("Random Patch!"))
            {
                std::lock_guard<std::mutex> paramLock(fm32.parameterMutex);
                // Randomize all!
                std::mt19937 rnd;
                rnd.seed(std::chrono::steady_clock::now().time_since_epoch().count());
                
                std::uniform_real_distribution<float> fdist(0.0f, 0.1f);
                auto frand = [&rnd, &fdist] { return fdist(rnd); };

                std::uniform_int_distribution<size_t> idist(1, 10);
                auto irand = [&rnd, &idist] { return idist(rnd); };

                for (size_t idx = 0; idx != fm32.parameterEnvelopes.freqMode.size(); ++idx)
                {
                    fm32.parameterEnvelopes.freqMode[idx] = rnd() & 1;
                }

                constexpr float releaseFudge = 8.0f;

                for (size_t idx = 0; idx != 1024; ++idx)
                {
                    fm32.parameterEnvelopes.modulationMatrix.attackDelta[idx] = frand();
                    fm32.parameterEnvelopes.modulationMatrix.attackOffset[idx] = frand();
                    fm32.parameterEnvelopes.modulationMatrix.decayDelta[idx] = frand();
                    fm32.parameterEnvelopes.modulationMatrix.decayOffset[idx] = frand();
                    fm32.parameterEnvelopes.modulationMatrix.delay[idx] = frand();
                    fm32.parameterEnvelopes.modulationMatrix.globalOffset[idx] = frand();
                    fm32.parameterEnvelopes.modulationMatrix.hold[idx] = frand();
                    fm32.parameterEnvelopes.modulationMatrix.releaseDelta[idx] = frand() * releaseFudge;
                    fm32.parameterEnvelopes.modulationMatrix.inverted[idx] = rnd() & 1;
                }

                for (size_t idx = 0; idx != 32; ++idx)
                {
                    fm32.parameterEnvelopes.operatorVolume.attackDelta[idx] = frand();
                    fm32.parameterEnvelopes.operatorVolume.attackOffset[idx] = frand();
                    fm32.parameterEnvelopes.operatorVolume.decayDelta[idx] = frand();
                    fm32.parameterEnvelopes.operatorVolume.decayOffset[idx] = frand();
                    fm32.parameterEnvelopes.operatorVolume.delay[idx] = frand();
                    fm32.parameterEnvelopes.operatorVolume.globalOffset[idx] = frand();
                    fm32.parameterEnvelopes.operatorVolume.hold[idx] = frand();
                    fm32.parameterEnvelopes.operatorVolume.releaseDelta[idx] = frand() * releaseFudge;
                    fm32.parameterEnvelopes.operatorVolume.inverted[idx] = rnd() & 1;
                }

                for (size_t idx = 0; idx != 32; ++idx)
                {
                    fm32.parameterEnvelopes.perOperatorFreq.attackDelta[idx] = frand();
                    fm32.parameterEnvelopes.perOperatorFreq.attackOffset[idx] = frand();
                    fm32.parameterEnvelopes.perOperatorFreq.decayDelta[idx] = frand();
                    fm32.parameterEnvelopes.perOperatorFreq.decayOffset[idx] = frand();
                    fm32.parameterEnvelopes.perOperatorFreq.delay[idx] = frand();
                    fm32.parameterEnvelopes.perOperatorFreq.globalOffset[idx] = frand() + irand();
                    fm32.parameterEnvelopes.perOperatorFreq.hold[idx] = frand();
                    fm32.parameterEnvelopes.perOperatorFreq.releaseDelta[idx] = frand() * releaseFudge;
                    fm32.parameterEnvelopes.perOperatorFreq.inverted[idx] = rnd() & 1;
                }

            }

            ImGui::Text("MIDI Devices");
            if (ImGui::Button("Refresh MIDI"))
            {
                static bool error = false;
                static std::string errorMessage;
                try
                {
                    error = false;
                    midiIn = std::make_unique<RtMidiIn>();
                }
                catch (RtMidiError& exc)
                {
                    errorMessage = exc.getMessage();
                    error = true;
                }

                if (error)
                {
                    ImGui::Text(errorMessage.c_str());
                }
            }

            
            static std::string midiDeviceName = "None";
            if (ImGui::BeginCombo("Device", midiDeviceName.c_str()))
            {
                static int selected;

                const auto nDevices = midiIn ? midiIn->getPortCount() : 0;
                if (nDevices)
                {
                    for (size_t idx = 0; idx != nDevices; ++idx)
                    {
                        // this thing throws!!
                        auto portName = midiIn->getPortName(idx);
                        if (ImGui::Selectable(portName.c_str(), selected == idx))
                        {
                            selected = idx;
                            midiIn->openPort(idx);
                        }
                    }
                }
                else {
                    ImGui::Selectable("None", true);
                    ImGui::SetItemDefaultFocus();
                }

                ImGui::EndCombo();
            }
    
            static size_t noteSlot;
            if (ImGui::Button("Press note"))
            {
                noteSlot = fm32.notePress(512.0f);
            }
            if (ImGui::Button("Release note"))
            {
                fm32.noteRelease(0);
            }


            //ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        } ImGui::End();


        // MIDI Handling

        if (midiIn)
        {
            std::vector<uint8_t> midiData;
            midiIn->getMessage(&midiData);
            auto messageIt = midiData.begin();

            //  x : MIDI Channel
            // 9x NOTE VEL : Note On message
            // 9x NOTE 00 : (Maschine) Note Off message
            // 8x NOTE 00 : Note Off message
            // Ax NOTE VEL: Aftertouch message
            // Bx CTR VAL: Control Change
            // Cx PRG: Program Change
            // Dx VAL: Pitch Bend

            while (messageIt != midiData.end())
            {
                switch((*messageIt++) >> 4)
                {
                case 0x8:
                    {
                        uint8_t note = *messageIt++;
                        /*uint8_t velocity = */messageIt++;
                        printf("Released note %d\n", note);
                        fm32.noteRelease(noteSlots[note]);
                        break;
                    }
                    case 0x9:
                    {
                        uint8_t note = *messageIt++;
                        uint8_t velocity = *messageIt++;
                        constexpr float baseNote = 440.0f;
                        if (velocity)
                        {
                            noteSlots[note] = fm32.notePress(
                                baseNote * std::pow(2.0f, (1.0f / 12.0f) * (int(note) - 57)));
                            printf("Pressed note %d with velocity %d (slot = %lld)\n", note, velocity, noteSlots[note]);
                        }
                        else
                        {
                            printf("Released note %d\n", note);
                            fm32.noteRelease(noteSlots[note]);
                        }
                        break;
                    }
                }
            }
        }
        
        // UI logic handling

        bool selectedCellChanged = false;
        static int selectedIdx = 0;
        const bool volumeSelected = selectedIdx & 0x10000;
        const bool freqSelected = selectedIdx & 0x20000;
        const int selectedX = selectedIdx & 31;
        const int selectedY = (selectedIdx >> 8) & 31;

        if (ImGui::Begin("Envelope Parameters"))
        {
            if (volumeSelected)
            {
                ImGui::Text("Operator %d Volume", selectedX + 1);
            }
            else if (freqSelected) {
                ImGui::Text("Operator %d frequency", selectedX + 1);
            }
            else { // Modulation Matrix selected
                if (selectedX == selectedY)
                {
                    ImGui::Text("Operator %d feedback", selectedX + 1);
                }
                else {
                    ImGui::Text("Operator %d on\nOperator %d", selectedY + 1, selectedX + 1);
                }
            }

            bool changed = false;
            
            if (volumeSelected)
            {
                const size_t volumeIdx = selectedX;
                auto& volumeVector = fm32.parameterEnvelopes.operatorVolume;

                changed |= ImGui::SliderFloat("Delay", &volumeVector.delay[volumeIdx], 0.0f, 3.0f);
                changed |= ImGui::SliderFloat("Global Offset", &volumeVector.globalOffset[volumeIdx], 0.0f, 3.0f);

                changed |= ImGui::SliderFloat("Attack Offset", &volumeVector.attackOffset[volumeIdx], 0.0f, 3.0f);
                changed |= ImGui::SliderFloat("Attack Delta", &volumeVector.attackDelta[volumeIdx], 0.0f, 3.0f);

                changed |= ImGui::SliderFloat("Hold", &volumeVector.hold[volumeIdx], 0.0f, 3.0f);

                changed |= ImGui::SliderFloat("Decay Offset", &volumeVector.decayOffset[volumeIdx], 0.0f, 3.0f);
                changed |= ImGui::SliderFloat("Decay Delta", &volumeVector.decayDelta[volumeIdx], 0.0f, 3.0f);

                changed |= ImGui::SliderFloat("Release Delta", &volumeVector.releaseDelta[volumeIdx], 0.0f, 3.0f);

                bool inverted = volumeVector.inverted[volumeIdx];
                if (ImGui::Checkbox("Inverted", &inverted))
                {
                    changed |= true;
                    volumeVector.inverted[volumeIdx] = inverted;
                }

                auto envelopeCurve = volumeVector.renderEnvelope<128>(volumeIdx, .0f, 0.1f);
                ImGui::PlotLines("Modulation Envelope", envelopeCurve.data(), envelopeCurve.size());
            }
            else if (freqSelected)
            {
                const size_t frequencyIdx = selectedX;
                auto& frequencyVector = fm32.parameterEnvelopes.perOperatorFreq;
                changed |= ImGui::SliderFloat("Delay", &frequencyVector.delay[frequencyIdx], 0.0f, 3.0f);

                {
                    int freqInt = std::floorf(frequencyVector.globalOffset[frequencyIdx]);
                    float freqFrac = frequencyVector.globalOffset[frequencyIdx] - freqInt;

                    if (ImGui::SliderInt("Global Offset Int", &freqInt, 0, 10)
                     || ImGui::SliderFloat("Global Offset Float", &freqFrac, 0.0f, 1.0f))
                    {
                        frequencyVector.globalOffset[frequencyIdx] = freqInt + freqFrac;
                        changed |= true;
                    }
                }

                changed |= ImGui::SliderFloat("Attack Offset", &frequencyVector.attackOffset[frequencyIdx], 0.0f, 3.0f);
                changed |= ImGui::SliderFloat("Attack Delta", &frequencyVector.attackDelta[frequencyIdx], 0.0f, 3.0f);

                changed |= ImGui::SliderFloat("Hold", &frequencyVector.hold[frequencyIdx], 0.0f, 3.0f);

                changed |= ImGui::SliderFloat("Decay Offset", &frequencyVector.decayOffset[frequencyIdx], 0.0f, 3.0f);
                changed |= ImGui::SliderFloat("Decay Delta", &frequencyVector.decayDelta[frequencyIdx], 0.0f, 3.0f);

                changed |= ImGui::SliderFloat("Release Delta", &frequencyVector.releaseDelta[frequencyIdx], 0.0f, 3.0f);

                bool inverted = frequencyVector.inverted[frequencyIdx];
                if (ImGui::Checkbox("Inverted", &inverted))
                {
                    changed |= true;
                    frequencyVector.inverted[frequencyIdx] = inverted;
                }

                bool relative = fm32.parameterEnvelopes.freqMode[frequencyIdx];
                if (ImGui::Checkbox("Relative", &relative))
                {
                    changed |= true;
                    fm32.parameterEnvelopes.freqMode[frequencyIdx] = relative;
                }

                auto envelopeCurve = frequencyVector.renderEnvelope<128>(frequencyIdx, .0f, 0.1f);
                ImGui::PlotLines("Modulation Envelope", envelopeCurve.data(), envelopeCurve.size());
            }
            else
            {
                const size_t envelopeIdx = selectedY * 32 + selectedX;
                auto& modMatrix = fm32.parameterEnvelopes.modulationMatrix;
                changed |= ImGui::SliderFloat("Delay", &modMatrix.delay[envelopeIdx], 0.0f, 3.0f);
                changed |= ImGui::SliderFloat("Global Offset", &modMatrix.globalOffset[envelopeIdx], 0.0f, 3.0f);

                changed |= ImGui::SliderFloat("Attack Offset", &modMatrix.attackOffset[envelopeIdx], 0.0f, 3.0f);
                changed |= ImGui::SliderFloat("Attack Delta", &modMatrix.attackDelta[envelopeIdx], 0.0f, 3.0f);

                changed |= ImGui::SliderFloat("Hold", &modMatrix.hold[envelopeIdx], 0.0f, 3.0f);

                changed |= ImGui::SliderFloat("Decay Offset", &modMatrix.decayOffset[envelopeIdx], 0.0f, 3.0f);
                changed |= ImGui::SliderFloat("Decay Delta", &modMatrix.decayDelta[envelopeIdx], 0.0f, 3.0f);

                changed |= ImGui::SliderFloat("Release Delta", &modMatrix.releaseDelta[envelopeIdx], 0.0f, 3.0f);

                bool inverted = modMatrix.inverted[envelopeIdx];
                if (ImGui::Checkbox("Inverted", &inverted))
                {
                    changed |= true;
                    modMatrix.inverted[envelopeIdx] = inverted;
                }

                auto envelopeCurve = modMatrix.renderEnvelope<128>(envelopeIdx, .0f, 0.1f);
                ImGui::PlotLines("Modulation Envelope", envelopeCurve.data(), envelopeCurve.size());
            }
        } ImGui::End();

        
        if (ImGui::Begin("Parameter Select"))
        {
            ImGui::Text("Modulation Matrix");
            
            for (int y = 0; y != 32; ++y)
            {
                ImGui::PushID(y << 8);
                    //ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.2f, 1.0f));
                        ImGui::RadioButton("", &selectedIdx, y << 8);
                    //ImGui::PopStyleColor();
                ImGui::PopID();

                for (int x = 1; x != 32; x++)
                {
                    ImGui::SameLine();
                    const int buttonId = y << 8 | x;
                    ImGui::PushID(buttonId);
                        ImGui::RadioButton("", &selectedIdx, buttonId);
                    ImGui::PopID();
                }
            }
            
            ImGui::Text("Volume Vector");
            ImGui::PushID(0x10000);
                ImGui::RadioButton("", &selectedIdx, 0x10000);
            ImGui::PopID();

            for (int x = 1; x != 32; x++)
            {
                ImGui::SameLine();
                ImGui::PushID(0x10000 + x);
                    ImGui::RadioButton("", &selectedIdx, 0x10000 + x);
                ImGui::PopID();
            }

            ImGui::Text("Frequency Vector");
            ImGui::PushID(0x20000);
                ImGui::RadioButton("", &selectedIdx, 0x20000);
            ImGui::PopID();

            for (int x = 1; x != 32; x++)
            {
                ImGui::SameLine();
                ImGui::PushID(0x20000 + x);
                    ImGui::RadioButton("", &selectedIdx, 0x20000 + x);
                ImGui::PopID();
            }

        } ImGui::End();

        // Rendering
        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

#else

	testFMSynthesis(32);
    testFMSynthesis(64);
    testFMSynthesis(128);
    testFMSynthesis(256);
    testFMSynthesis(512);
    testFMSynthesis(1024);

#endif

#ifndef BENCHMARK
    //SDL_PauseAudio(true);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
#endif

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

    cudaDeviceSynchronize();

	return 0;
}
