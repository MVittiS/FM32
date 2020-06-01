#include "FM32Kernel.cuh"

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <cassert>

__device__ __host__ inline float mix(const float mixFactor, const float a, const float b)
{
    return (a * (1.0f - mixFactor)) + (b * mixFactor);
}

__device__ __host__ inline float fract(const float in)
{
    return in - truncf(in);
}

__global__ void reduceFM(const float* __restrict__ voicesBuffer, float* __restrict__ outputBuffer,
    size_t nVoices, size_t nActiveVoices, size_t nSamples, size_t nChannels)
{
    size_t idx = threadIdx.x;
    for (size_t sample = 0; sample != nSamples; ++sample)
    {
        float voicesAccum = idx < nActiveVoices ? voicesBuffer[sample * nVoices + idx] : .0f;

        constexpr uint32_t ALL_WARPS = 0xFFFFFFFF;
        voicesAccum += __shfl_xor_sync(ALL_WARPS, voicesAccum, 1, 32);
        voicesAccum += __shfl_xor_sync(ALL_WARPS, voicesAccum, 2, 32);
        voicesAccum += __shfl_xor_sync(ALL_WARPS, voicesAccum, 4, 32);
        voicesAccum += __shfl_xor_sync(ALL_WARPS, voicesAccum, 8, 32);
        voicesAccum += __shfl_xor_sync(ALL_WARPS, voicesAccum, 16, 32);

        if (idx < nChannels)
        {
            outputBuffer[nChannels * sample + idx] = voicesAccum;
        }
    }

}

void testFMSynthesis(size_t samplesPerBlock)
{
    constexpr size_t numChannels = 2;
    constexpr size_t numActiveVoices = 32;

    CPUOperatorPack<32, numActiveVoices> ops;

    std::cout << "Size of FM Operator packs: " 
        << sizeof(GPUOperatorPack<32>) << "b per voice." << std::endl;

    auto& gpuOps = ops.cpuOperators;
    constexpr float samplingRate = 48000.0f;
    constexpr float noteFreq = 512.0f;
    const size_t nBlocks = (1024 * 1024 * 8) / samplesPerBlock;

    const float noteIncrement = noteFreq / samplingRate;
    gpuOps.stateBegin.phaseIncrement[0] = gpuOps.stateEnd.phaseIncrement[0] = 3.0f * noteIncrement;
    gpuOps.stateBegin.phaseIncrement[1] = gpuOps.stateEnd.phaseIncrement[1] = noteIncrement;
    gpuOps.stateBegin.outputVolume[0] = gpuOps.stateEnd.outputVolume[0] = 0.0f;
    gpuOps.stateBegin.outputVolume[1] = gpuOps.stateEnd.outputVolume[1] = 0.7f;
    ops.updateGPUOperators();


    const size_t blockSizeInBytes = sizeof(float) * samplesPerBlock * numChannels;

    float* cpuBuffer = new float[samplesPerBlock * nBlocks * numChannels];
    std::fill(cpuBuffer, cpuBuffer + samplesPerBlock, 0.0f);
    float* gpuAudioBuffer, *gpuIntermediateBuffer;
    cudaMalloc(&gpuAudioBuffer, blockSizeInBytes);
    cudaMemset(gpuAudioBuffer, 0, blockSizeInBytes);
    cudaMalloc(&gpuIntermediateBuffer, sizeof(float) * samplesPerBlock * 32);

    std::chrono::steady_clock clock;
    auto timeStart = clock.now();
    std::vector<decltype(clock.now() - timeStart)> latencies(nBlocks);

    for (auto x = 0; x != nBlocks; ++x)
    {
        const auto begin = clock.now();
        gpuOps.stateBegin.modMatrix[0][1] = 1.5f * std::expf(-x * 0.02f);
        gpuOps.stateEnd.modMatrix[0][1] = 1.5f * std::expf(-(x + 1) * 0.02f);
        ops.updateGPUOperators();
        
        cudaMemcpy(cpuBuffer + x * samplesPerBlock * numChannels, gpuAudioBuffer,
            blockSizeInBytes, ::cudaMemcpyDeviceToHost);
        processFM<32><<<32, 32>>>(ops.gpuOperators, sizeof(GPUOperatorPack<32>), 
            numActiveVoices, samplesPerBlock, gpuIntermediateBuffer);
        reduceFM << <32, 1 >> > (gpuIntermediateBuffer, gpuAudioBuffer, 32,
            numActiveVoices, samplesPerBlock, numChannels);
        
        const auto end = clock.now();
        latencies[x] = (end - begin);
    }

    auto timeEnd = clock.now();
    auto timeElapsed = timeEnd - timeStart;


    decltype(timeElapsed) timeMedian, time95, time999;
    {
        decltype(latencies) latenciesSorted(latencies.begin()/* + 100*/, latencies.end());
        std::sort(latenciesSorted.begin(), latenciesSorted.end());
        const auto nPoints = latenciesSorted.size();
        timeMedian = latenciesSorted[nPoints / 2] / 1000;
        time95 = latenciesSorted[(nPoints * 19) / 20] / 1000;
        time999 = latenciesSorted[(nPoints * 999) / 1000] / 1000;
    }

    std::cout << "Generating 1M FM samples (buffer size = "
        << samplesPerBlock << ") took " << timeElapsed.count() / 1'000'000 << "ms.\n";
    std::cout << "Median: " << timeMedian.count()
        << "us, 95%: " << time95.count() << "us, 99.9%: " << time999.count() << "us\n";

    std::ofstream outFile("FMout.bin", std::ios::binary);
    outFile.write(reinterpret_cast<char*>(cpuBuffer), sizeof(float) * samplesPerBlock * nBlocks);
    outFile.close();

    //cudaFreeHost(cpuBuffer);
    cudaFree(gpuIntermediateBuffer);
    cudaFree(gpuAudioBuffer);
    delete[] cpuBuffer;
}

void testFMSynthesis_CPU(size_t samplesPerBlock)
{
    CPUOperatorPack<32> ops;

    std::cout << "Size of FM Operator packs: "
        << sizeof(GPUOperatorPack<32>) << "b per voice." << std::endl;

    auto& gpuOps = ops.cpuOperators;
    constexpr float samplingRate = 48000.0f;
    constexpr float noteFreq = 512.0f;
    const size_t nBlocks = (1024 * 1024 * 8) / samplesPerBlock;

    const float noteIncrement = noteFreq / samplingRate;
    gpuOps.stateBegin.phaseIncrement[0] = gpuOps.stateEnd.phaseIncrement[0] = 3.0f * noteIncrement;
    gpuOps.stateBegin.phaseIncrement[1] = gpuOps.stateEnd.phaseIncrement[1] = noteIncrement;
    gpuOps.stateBegin.outputVolume[0] = gpuOps.stateEnd.outputVolume[0] = 0.0f;
    gpuOps.stateBegin.outputVolume[1] = gpuOps.stateEnd.outputVolume[1] = 0.7f;

    constexpr size_t numChannels = 2;
    constexpr size_t numActiveVoices = 32;

    float* cpuBuffer = new float[samplesPerBlock * nBlocks * numChannels];

    std::chrono::steady_clock clock;
    auto timeStart = clock.now();
    std::vector<decltype(clock.now() - timeStart)> latencies(nBlocks);

    for (auto x = 0; x != nBlocks; ++x)
    {
        auto begin = clock.now();
        gpuOps.stateBegin.modMatrix[0][1] = 1.5f * std::expf(-x * 0.02f);
        gpuOps.stateEnd.modMatrix[0][1] = 1.5f * std::expf(-(x + 1) * 0.02f);

        GPUOperatorPack<32> parameters[32];
        for (auto& pack : parameters)
        {
            pack = ops.cpuOperators;
        }

        processFM_CPU<32>(parameters, numActiveVoices,
            samplesPerBlock, numChannels, cpuBuffer);
        auto end = clock.now();
        latencies[x] = (end - begin);
    }

    auto timeEnd = clock.now();
    auto timeElapsed = timeEnd - timeStart;


    decltype(timeElapsed) timeMedian, time95, time999;
    {
        decltype(latencies) latenciesSorted(latencies.begin()/* + 100*/, latencies.end());
        std::sort(latenciesSorted.begin(), latenciesSorted.end());
        auto nPoints = latenciesSorted.size();
        timeMedian = latenciesSorted[nPoints / 2] / 1000;
        time95 = latenciesSorted[(nPoints * 19) / 20] / 1000;
        time999 = latenciesSorted[(nPoints * 999) / 1000] / 1000;
    }

    std::cout << "Generating 1M FM samples (buffer size = "
        << samplesPerBlock << ") took " << timeElapsed.count() / 1000000 << "ms.\n";
    std::cout << "Median: " << timeMedian.count()
        << "us, 95%: " << time95.count() << "us, 99.9%: " << time999.count() << "us\n";

    std::ofstream outFile("FMout.bin", std::ios::binary);
    outFile.write(reinterpret_cast<char*>(cpuBuffer), sizeof(float)* samplesPerBlock* nBlocks);
    outFile.close();

    delete[] cpuBuffer;
}

void TestReduceFM() {
    std::array<float, 32 * 128> matrixIn;
    std::array<float, 128> vectorOut;

    // Diagonal matrix
    for (auto idx = 0; idx != 32; ++idx) {
        for (auto subIdx = 0; subIdx != 32; ++subIdx) {
            auto offset = 32 * idx + subIdx;
            matrixIn[offset] = (idx == subIdx) ? idx : 0;
        }
        vectorOut[idx] = idx;
    }

    // Flipped diagonal matrix
    for (auto idx = 0; idx != 32; ++idx) {
        for (auto subIdx = 0; subIdx != 32; ++subIdx) {
            auto offset = 32 * (idx + 32) + subIdx;
            matrixIn[offset] = (idx == (31 - subIdx)) ? idx : 0;
        }
        vectorOut[idx + 32] = idx;
    }

    // Stripped Matrix
    for (auto idx = 0; idx != 32; ++idx) {
        for (auto subIdx = 0; subIdx != 32; ++subIdx) {
            auto offset = 32 * (idx + 64) + subIdx;
            matrixIn[offset] = ((idx / 2) == ((15 - subIdx) & 15)) ? idx : 0;
        }
        vectorOut[idx + 64] = 2 * idx;
    }

    // Helical Scan Matrix
    for (auto idx = 0; idx != 32; ++idx) {
        for (auto subIdx = 0; subIdx != 32; ++subIdx) {
            auto offset = 32 * (idx + 96) + subIdx;
            matrixIn[offset] = (idx + subIdx) & 31;
        }
        vectorOut[idx + 96] = 31 * 16;
    }

    float *gpuMatrixIn, *gpuVectorOut;
    cudaMalloc(&gpuMatrixIn, sizeof(float) * matrixIn.size());
    cudaMemcpy(gpuMatrixIn, matrixIn.data(), sizeof(float) * matrixIn.size(),
               ::cudaMemcpyHostToDevice);
    cudaMalloc(&gpuVectorOut, sizeof(float) * vectorOut.size());

    const size_t nVoices = 32;
    const size_t nActiveVoices = 32;
    const size_t nSamples = matrixIn.size() / nVoices;
    const size_t nChannels = 1;
    reduceFM<<<1, 32>>>(gpuMatrixIn, gpuVectorOut, nVoices,
        nActiveVoices, nSamples, nChannels);

    std::array<float, 128> gpuResult;
    cudaMemcpy(gpuResult.data(), gpuVectorOut, sizeof(float) * vectorOut.size(),
               ::cudaMemcpyDeviceToHost);

    if (gpuResult != vectorOut) {
        printf("Error: reduceFM results differ from expectation.\n");
        // Print both vectors size-by-side
        for (auto idx = 0; idx != vectorOut.size(); ++idx) {
            printf("%4.4f   %4.4f\n", vectorOut[idx], gpuResult[idx]);
        }
    }
    else {
        printf("ReduceFM is working properly.\n");
    }
}


FMSynth::FMSynth(size_t sampleRate, size_t samplesPerFrame, size_t channels)
: sampleRate(sampleRate)
, samplesPerFrame(samplesPerFrame)
, invSampleRate(1.0f / sampleRate)
, nChannels(channels)
{
    inUse.reset();
    cudaMallocPitch(&gpuOps, &allocPitch,
        sizeof(GPUOperatorPack<32>), FMVoices);
    
    cudaMalloc(&gpuOperatorBuffer, sizeof(float) * samplesPerFrame);
    cudaMemset(gpuOperatorBuffer, 0, samplesPerFrame);

    parameterEnvelopes.freqMode.reset();
    parameterEnvelopes.perOperatorFreq.clear();
    parameterEnvelopes.modulationMatrix.clear();
    parameterEnvelopes.operatorVolume.clear();
}

FMSynth::~FMSynth()
{
    std::lock_guard<std::mutex> lockMyself(parameterMutex);
    for (size_t x = 0; x != 32; ++x)
    {
        cudaFree(gpuOps);
        cudaFree(gpuOperatorBuffer);
    }
    //std::ofstream audioDumpFile("AudioDump.bin");
    //audioDumpFile.write((char *)audioDump.data(), sizeof(float) * audioDump.size());
}

void FMSynth::setSampleRate(size_t newSampleRate)
{
    std::lock_guard<std::mutex> lockMyself(parameterMutex);
    sampleRate = newSampleRate;
    invSampleRate = 1.0f / newSampleRate;
}

void FMSynth::setSamplesPerFrame(size_t newSamplesPerFrame)
{
    std::lock_guard<std::mutex> lockMyself(parameterMutex);
    cudaFree(gpuOperatorBuffer);
    cudaFree(gpuAudioBuffer);
    cudaMalloc(&gpuOperatorBuffer, sizeof(float) * FMVoices * newSamplesPerFrame);
    cudaMalloc(&gpuAudioBuffer, sizeof(float) * newSamplesPerFrame * nChannels);
    samplesPerFrame = newSamplesPerFrame;
}

void FMSynth::setNumChannels(size_t newNumChannels)
{
    std::lock_guard<std::mutex> lockMyself(parameterMutex);
    cudaFree(gpuAudioBuffer);
    cudaMalloc(&gpuAudioBuffer, sizeof(float) * samplesPerFrame * newNumChannels);
    nChannels = newNumChannels;
}

static constexpr size_t noFreeSlot = std::numeric_limits<size_t>::max();

size_t FMSynth::notePress(float freq)
{
    recycleNoteSlots();

    size_t idx;
    for (idx = 0; idx != inUse.size(); ++idx)
    {
        if (!inUse[idx])
        {
            inUse[idx] = true;
            released[idx] = false;

            notesBaseFrequency[idx] = freq;
            noteStartCycle[idx] = currentCycle;

            //printf("Allocated note at slot %d with frequency %f.\n", idx, freq);
            //printf("Note usage bitmap: %s\n", inUse.to_string('0', '1').c_str());

            return idx;
        }
    }
    return noFreeSlot;
}

void FMSynth::noteRelease(size_t slot)
{
    if (slot == noFreeSlot || released[slot])
    {
        return;
    }
    const float releaseTime = (currentCycle - noteStartCycle[slot]) * invSampleRate;
    releasedModMatrices[slot] = parameterEnvelopes.modulationMatrix.releaseOnTime(releaseTime);
    releasedVolumes[slot] = parameterEnvelopes.operatorVolume.releaseOnTime(releaseTime);
    releasedFrequencies[slot] = parameterEnvelopes.perOperatorFreq.releaseOnTime(releaseTime);
    //printf("Releasing note at slot %d with frequency %f.\n", slot, notesBaseFrequency[slot]);

    released[slot] = true;
}

void FMSynth::noteKill(size_t slot)
{
    inUse[slot] = false;
}

void FMSynth::recycleNoteSlots()
{
    for (size_t idx = 0; idx != released.size(); idx++)
    {
        if (!released[idx])
        {
            continue;
        }

        const float time = (currentCycle - noteStartCycle[idx]) * invSampleRate;

        const auto outputVolumes = parameterEnvelopes.operatorVolume
            .forTimeOnRelease(time, releasedVolumes[idx]);

        //const float sum = std::reduce(outputVolumes.begin(), outputVolumes.end()); // CUDA is not yet C++17...
        const float sum = std::accumulate(outputVolumes.begin(), outputVolumes.end(), .0f);
        constexpr float tooQuiet = 1e-6f;
        if (std::abs(sum) < tooQuiet)
        {
            noteKill(idx);
        }
    }
}

void FMSynth::renderAudio()
{
    struct PackedFMState
    {
        FMState<32> stateBegin, stateEnd;
    };
    std::vector<PackedFMState> packedFMState(0);

    for (size_t slot = 0; slot != FMVoices; ++slot)
    {
        if (!inUse[slot])
        {
            continue;
        }

        PackedFMState slotState;

        const float noteTimeBegin = 
            (currentCycle - noteStartCycle[slot]) * invSampleRate;
        const float noteTimeEnd =
            (currentCycle - noteStartCycle[slot] + samplesPerFrame) * invSampleRate;

        const auto opFrequenciesBegin = released[slot] ?
            parameterEnvelopes.perOperatorFreq.forTimeOnRelease(noteTimeBegin, releasedFrequencies[slot])
            : parameterEnvelopes.perOperatorFreq.forTime(noteTimeBegin);

        const auto opFrequenciesEnd = released[slot] ?
            parameterEnvelopes.perOperatorFreq.forTimeOnRelease(noteTimeEnd, releasedFrequencies[slot])
            : parameterEnvelopes.perOperatorFreq.forTime(noteTimeEnd);

        const float noteResolvedFrequency = 
            (parameterEnvelopes.freqMode == SynthParameterPack::FreqMode::Fixed) 
            ? 1.0f 
            : notesBaseFrequency[slot];

        for (size_t idx = 0; idx != opFrequenciesBegin.size(); ++idx)
        {
            const float phaseIncrement = opFrequenciesBegin[idx] * noteResolvedFrequency;
            slotState.stateBegin.phaseIncrement[idx] = phaseIncrement * invSampleRate;
        }
        for (size_t idx = 0; idx != opFrequenciesEnd.size(); ++idx)
        {
            const float phaseIncrement = opFrequenciesEnd[idx] * noteResolvedFrequency;
            slotState.stateEnd.phaseIncrement[idx] = phaseIncrement * invSampleRate;
        }

        const auto modulationMatrixBegin = released[slot] ? 
            parameterEnvelopes.modulationMatrix.forTimeOnRelease(noteTimeBegin, releasedModMatrices[slot])
            : parameterEnvelopes.modulationMatrix.forTime(noteTimeBegin);

        const auto modulationMatrixEnd = released[slot] ? 
            parameterEnvelopes.modulationMatrix.forTimeOnRelease(noteTimeEnd, releasedModMatrices[slot])
            : parameterEnvelopes.modulationMatrix.forTime(noteTimeEnd);

        memcpy(*slotState.stateBegin.modMatrix, modulationMatrixBegin.data(),
            sizeof(FMState<32>::modMatrix));
        memcpy(*slotState.stateEnd.modMatrix, modulationMatrixEnd.data(),
            sizeof(FMState<32>::modMatrix));

        const auto opVolumeBegin = released[slot] ? 
            parameterEnvelopes.operatorVolume.forTimeOnRelease(noteTimeBegin, releasedVolumes[slot])
            : parameterEnvelopes.operatorVolume.forTime(noteTimeBegin);

        const auto opVolumeEnd = released[slot] ? 
            parameterEnvelopes.operatorVolume.forTimeOnRelease(noteTimeEnd, releasedVolumes[slot])
            : parameterEnvelopes.operatorVolume.forTime(noteTimeEnd);

        memcpy(slotState.stateBegin.outputVolume, opVolumeBegin.data(),
            sizeof(FMState<32>::outputVolume));
        memcpy(slotState.stateEnd.outputVolume, opVolumeEnd.data(),
            sizeof(FMState<32>::outputVolume));

        packedFMState.emplace_back(std::move(slotState));
    }

    if (cudaMemcpy2D(gpuOps, allocPitch,
        packedFMState.data(), sizeof(packedFMState[0]),
        sizeof(PackedFMState), packedFMState.size(),
        cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("Error during cudaMemcpy2D!!\n");
    }

    //printf("Notes being rendered: %d\n", packedFMState.size()); WORKS
    processFM<<<32, 32>>>(gpuOps, allocPitch, packedFMState.size(), samplesPerFrame, gpuOperatorBuffer);
    reduceFM<<<1, 32>>>(gpuOperatorBuffer, gpuAudioBuffer, FMVoices, 
        packedFMState.size(), samplesPerFrame, nChannels);
    currentCycle += samplesPerFrame;
}


void FMSynth::renderAudioDebug()
{
    std::array<float, 32> frequencyMult;
    for (size_t idx = 0; idx != frequencyMult.size(); ++idx)
    {
        frequencyMult[idx] = idx;
    }

    struct PackedFMState
    {
        FMState<32> stateBegin, stateEnd;
    };
    std::vector<PackedFMState> packedFMState;

    for (size_t slot = 0; slot != FMVoices; ++slot)
    {
        if (!inUse[slot])
        {
            continue;
        }

        packedFMState.emplace_back();

        for (size_t idx = 0; idx != 32; ++idx)
        {
            packedFMState.back().stateBegin.phaseIncrement[idx] = slot * 100.0f + idx;
        }
        for (size_t idx = 0; idx != 32; ++idx)
        {
            packedFMState.back().stateEnd.phaseIncrement[idx] = slot * 100.0f + idx;
        }

        std::array<float, 1024> modulationMatrix;
        std::fill(modulationMatrix.begin(), modulationMatrix.end(), slot);

        memcpy(*packedFMState.back().stateBegin.modMatrix, modulationMatrix.data(),
            sizeof(FMState<32>::modMatrix));
        memcpy(*packedFMState.back().stateEnd.modMatrix, modulationMatrix.data(),
            sizeof(FMState<32>::modMatrix));

        std::array<float, 32> opVolume;
        std::fill(opVolume.begin(), opVolume.end(), slot);

        memcpy(packedFMState.back().stateBegin.outputVolume, opVolume.data(),
            sizeof(FMState<32>::outputVolume));
        memcpy(packedFMState.back().stateEnd.outputVolume, opVolume.data(),
            sizeof(FMState<32>::outputVolume));
    }

    if (cudaMemcpy2D(gpuOps, allocPitch,
        packedFMState.data(), sizeof(packedFMState[0]),
        sizeof(PackedFMState), packedFMState.size(),
        cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("Error during cudaMemcpy2D!!\n");
    }
    processFM<<<32, 32>>> (gpuOps, allocPitch, packedFMState.size(), samplesPerFrame, gpuOperatorBuffer);
    reduceFM<<<32, 1>>> (gpuOperatorBuffer, gpuAudioBuffer, FMVoices,
        packedFMState.size(), samplesPerFrame, nChannels);
    currentCycle += samplesPerFrame;
}


void FMSynth::getRenderedAudio(float* outputBuffer)
{
    cudaMemcpy(outputBuffer, gpuAudioBuffer,
        sizeof(float) * samplesPerFrame * nChannels, cudaMemcpyDeviceToHost);

    //audioDump.insert(audioDump.end(), outputBuffer, outputBuffer + samplesPerFrame * nChannels);
}

void FMSynth::allInUse()
{
    for (auto x = 0; x != inUse.size(); ++x) {
        inUse[x] = true;
    }
}
