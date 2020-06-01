#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cassert>
#define M_PI       3.14159265358979323846

#include <algorithm>
#include <array>
#include <bitset>
#include <future>
#include <numeric>
#include <mutex>


constexpr size_t MaxVoices = 32;

__device__ __host__ inline float mix(const float mixFactor, const float a, const float b);
__device__ __host__ inline float fract(const float in);


template <int N>
struct FMState
{
    float phaseIncrement[N];
    float modMatrix[N][N];
    float outputVolume[N];
};

template <int N>
struct GPUOperatorPack
{
    FMState<N> stateBegin, stateEnd;
	float previousOutput[N];
    float phase[N];
};

template <int N, int nVoices = 1>
struct CPUOperatorPack
{
	GPUOperatorPack<N> cpuOperators;
	GPUOperatorPack<N>* gpuOperators;
    size_t allocPitch;

	void updateGPUOperatorFreqs()
	{
		cudaMemcpy
		(
			gpuOperators->stateBegin.phaseIncrement,
			cpuOperators.stateBegin.phaseIncrement,
			sizeof(GPUOperatorPack<N>::phaseIncrement),
			::cudaMemcpyHostToDevice
		);
        cudaMemcpy
        (
            gpuOperators->stateEnd.phaseIncrement,
            cpuOperators.stateEnd.phaseIncrement,
            sizeof(GPUOperatorPack<N>::phaseIncrement),
            ::cudaMemcpyHostToDevice
        );
	};

	void updateGPUModulationMatrix()
	{
        cudaMemcpy
        (
            gpuOperators->stateBegin.modMatrix[0],
            cpuOperators.stateBegin.modMatrix[0],
            sizeof(GPUOperatorPack<N>::modMatrix),
            ::cudaMemcpyHostToDevice
        );
		cudaMemcpy
		(
            gpuOperators->stateEnd.modMatrix[0],
            cpuOperators.stateEnd.modMatrix[0],
			sizeof(GPUOperatorPack<N>::modMatrix),
			::cudaMemcpyHostToDevice
		);
	};

	void updateGPUOperatorVolumes()
	{
		cudaMemcpy
		(
			gpuOperators->stateBegin.outputVolume,
			cpuOperators.stateBegin.outputVolume,
			sizeof(GPUOperatorPack<N>::outputVolume),
			::cudaMemcpyHostToDevice
		);
        cudaMemcpy
        (
            gpuOperators->stateEnd.outputVolume,
            cpuOperators.stateEnd.outputVolume,
            sizeof(GPUOperatorPack<N>::outputVolume),
            ::cudaMemcpyHostToDevice
        );
	};

	void updateGPUOperators()
	{
		cudaMemcpy
		(
			gpuOperators,
			&cpuOperators,
			sizeof(FMState<N>) * 2,
			::cudaMemcpyHostToDevice
		);

        for (size_t voice = 1; voice != nVoices; ++voice)
        {
            cudaMemcpy
            (
                gpuOperators + allocPitch * voice,
                gpuOperators,
                sizeof(FMState<N>) * 2,
                ::cudaMemcpyDeviceToDevice
            );
        }
	};

	CPUOperatorPack()
	{
        memset(&cpuOperators, 0, sizeof(GPUOperatorPack<N>));
        cudaMallocPitch(&gpuOperators, &allocPitch, sizeof(GPUOperatorPack<N>), nVoices);
		updateGPUOperators();
	};

	~CPUOperatorPack()
	{
		cudaFree(gpuOperators);
	};

};

template <int N>
__global__ void processFM(GPUOperatorPack<N>* operators, size_t allocPitch,
	const size_t nVoices, const size_t nSamples, float* outputBuffer)
{
    const float invNSamples = 1.0f / float(nSamples);
    constexpr float twoPi = M_PI * 2.0f;
	const auto voice = blockIdx.x;
	const auto op = threadIdx.x;

    if (op >= N || voice >= nVoices)
    {
        return;
    }

    GPUOperatorPack<N>& thisOperator = *(GPUOperatorPack<N>*)
        ((char*)operators + (allocPitch * voice));

    __shared__ float previousPhase[N];
    previousPhase[op] = thisOperator.phase[op];
    
    __shared__ float previousOutput[N];
    previousOutput[op] = thisOperator.previousOutput[op];

    for (int sample = 0; sample < nSamples; sample++)
	{
        const float mixFactor = sample * invNSamples;
        const float oldPhase = previousPhase[op];

		// Each thread is an Operator;
        // Each block a Voice!
		float fmAccum = 0.0f;

		for (int x = 0; x < N; x++) 
        {
			fmAccum += previousOutput[x]
				* mix(mixFactor, 
                      thisOperator.stateBegin.modMatrix[x][op], // Invert the access dimension, and you get a 10x slowdown on a GT640M -
                      thisOperator.stateEnd.modMatrix[x][op]);  // but only a 30% slowdown on a P400; likely due to L2 cache size!
		}

        const float fmOutput = __sinf(twoPi * (oldPhase + fmAccum));
        previousOutput[op] = fmOutput;

        const float phaseIncrement = mix(mixFactor,
            thisOperator.stateBegin.phaseIncrement[op],
			thisOperator.stateEnd.phaseIncrement[op]);

        previousPhase[op] = fract(oldPhase + phaseIncrement);

        const float mixedState = fmOutput * mix(mixFactor,
            thisOperator.stateBegin.outputVolume[op], thisOperator.stateEnd.outputVolume[op]);

		float accum = mixedState;
        constexpr uint32_t ALL_WARPS = 0xFFFFFFFF;
		accum += __shfl_xor_sync(ALL_WARPS, accum, 1, 32);
		accum += __shfl_xor_sync(ALL_WARPS, accum, 2, 32);
		accum += __shfl_xor_sync(ALL_WARPS, accum, 4, 32);
		accum += __shfl_xor_sync(ALL_WARPS, accum, 8, 32);
		accum += __shfl_xor_sync(ALL_WARPS, accum, 16, 32);

		if (op == 0) {
            outputBuffer[N * sample + voice] = accum;
		}
	}

    thisOperator.phase[op] = previousPhase[op];
    thisOperator.previousOutput[op] = previousOutput[op];
}

// Only works in the scope of a single block!!
__global__ void reduceFM(const float* voicesBuffer, float* outputBuffer, size_t nVoices,
    size_t nActiveVoices, size_t nSamples, size_t nChannels);


void TestReduceFM();


// CPU Implementation, for comparison
template <int N>
void processFM_CPU(GPUOperatorPack<N>* operators, const size_t nVoices,
    const size_t nSamples, size_t nChannels, float* outputBuffer)
{
    const float invNSamples = 1.0f / float(nSamples);
    constexpr float twoPi = float(M_PI * 2.0);

    for (int sample = 0; sample < nSamples; sample++)
    {
        std::vector<float> voiceOutput(nVoices);

        for (size_t voice = 0; voice != nVoices; ++voice)
        {
            GPUOperatorPack<N>& thisOperator = operators[voice];
            auto newPhase = thisOperator.phase;
            auto newOutput = thisOperator.previousOutput;

            std::array<float, N> operatorOutput;
            for (size_t op = 0; op != N; ++op)
            {
                const float mixFactor = sample * invNSamples;
                const float oldPhase = thisOperator.phase[op];

                float fmAccum = 0.0f;

                for (int x = 0; x < N; x++)
                {
                    fmAccum += thisOperator.previousOutput[x]
                        * mix(mixFactor,
                            thisOperator.stateBegin.modMatrix[op][x], // Transposed in relation to GPU!!
                            thisOperator.stateEnd.modMatrix[op][x]); 
                }

                const float fmOutput = sinf(twoPi * (oldPhase + fmAccum));
                newOutput[op] = fmOutput;

                const float phaseIncrement = mix(mixFactor,
                    thisOperator.stateBegin.phaseIncrement[op],
                    thisOperator.stateEnd.phaseIncrement[op]);

                newPhase[op] = fract(oldPhase + phaseIncrement);

                const float mixedState = fmOutput * mix(mixFactor,
                    thisOperator.stateBegin.outputVolume[op], thisOperator.stateEnd.outputVolume[op]);

                operatorOutput[op] = mixedState;
            }

            voiceOutput[voice] = std::accumulate(operatorOutput.begin(),
                                                 operatorOutput.end(), 0.0f);
            
            std::copy_n(newOutput, N, thisOperator.previousOutput);
            std::copy_n(newPhase, N, thisOperator.phase);
        }

        const float sampleOutput = std::accumulate(voiceOutput.begin(),
                                                   voiceOutput.end(), 0.0f);
        for (size_t ch = 0; ch != nChannels; ++ch)
        {
            outputBuffer[nChannels * sample + ch] = sampleOutput;
        }
    }
}


/*
  To make it easier to compute, the envelope is calculated purely using a math formula 
  without branching - so we could even move it to the GPU, perhaps.

  Some parameters are automatically scaled by the inverse of the sample rate as soon as 
  they're received (namely, deltas), so you can change sample rates on the fly without
  having to worry about them.

  = globalOffset =
    Offset that's applied to the envelope, after all other calculations have been done.
    Equivalent to a DC bias.

  = delay =
    Time in seconds before the envelope is started.

  = attackOffset =
    Constant factor added to attack, when started.

  = attackDelta = 
    The tangent (or rate of change) that will be added per *second* to the parameter.

  = hold =
    Limiting constant - as soon as the current parameter reaches *hold*, it won't grow
    or reduce anymore and remain stable.

  = decayOffset =
    Starting point to which the decay calculations are based from.

  = decayDelta =
    Tangent (or rate of change) to base decay from.

  = releaseDelta =
    Tangent (or rate of change) to base release from. The offset is internally calculated
    by back-projecting it to the current parameter value at the time a release is requested.

    Formula (Desmos/LaTeX):
    g_{off}+\max\left(\min\left(\min\left(a_{off}+a_{del}\cdot\left(x-d\right),h\right),\ d_{off}-d_{del}\cdot\left(x-d\right)\right),\ 0\right)
*/


template <int N>
struct Envelope
{
    float globalOffset[N],
          delay[N],
          attackOffset[N],
          attackDelta[N],
          hold[N],
          decayOffset[N],
          decayDelta[N],
          releaseDelta[N];
    std::bitset<N> inverted;

    void clear()
    {
        std::fill_n(globalOffset, N, .0f);
        std::fill_n(delay, N, .0f);
        std::fill_n(attackOffset, N, .0f);
        std::fill_n(attackDelta, N, .0f);
        std::fill_n(hold, N, .0f);
        std::fill_n(decayOffset, N, .0f);
        std::fill_n(decayDelta, N, .0f);
        std::fill_n(releaseDelta, N, .0f);

        inverted.reset();
    }

    std::array<float, N> forTime(float t) const
    {
        std::array<float, N> results;
        for (size_t idx = 0; idx != N; ++idx)
        {
            const float tOffset = t - delay[idx];
            
            const float tForTime = std::max(
                std::min(std::min(
                    attackOffset[idx] + attackDelta[idx] * tOffset,
                    hold[idx]),
                    decayOffset[idx] - decayDelta[idx] * tOffset),
                .0f);
            const float tDelayed = tOffset > 0.0f ? tForTime : 0.0f;
            results[idx] = (inverted[idx] ? -tDelayed : tDelayed) + globalOffset[idx];
        }
        return results;
    };

    [[nodiscard]] const std::array<float, N> releaseOnTime(float t) const
    {
        std::array<float, N> releaseOffset, envelopeForT {this->forTime(t)};
        for (size_t idx = 0; idx != N; ++idx)
        {
            releaseOffset[idx] = (t * releaseDelta[idx]) + 
                (envelopeForT[idx] - globalOffset[idx]);
        }
        return releaseOffset;
    }

    std::array<float, N> forTimeOnRelease(float t, const std::array<float, N>& releaseOffset) const
    {
        std::array<float, N> results;
        for (size_t idx = 0; idx != N; ++idx)
        {
            const float tForTime = std::max(releaseOffset[idx] - releaseDelta[idx] * t, .0f);
            results[idx] = (inverted[idx] ? -tForTime : tForTime) + globalOffset[idx];
        }
        return results;
    }

    template <int NumOut>
    std::array<float, NumOut> renderEnvelope(size_t idx, float tBegin, float tDelta)
    {
        const Envelope<1> params =
        {
            globalOffset[idx],
            delay[idx],
            attackOffset[idx],
            attackDelta[idx],
            hold[idx],
            decayOffset[idx],
            decayDelta[idx],
            releaseDelta[idx]
        };

        std::array<float, NumOut> envelopeCurve;
        for (size_t x = 0; x != NumOut; ++x)
        {
            envelopeCurve[x] = params.forTime(tBegin + (tDelta * x))[0];
        }

        return envelopeCurve;
    }
};

constexpr size_t FMVoices = 32;

struct SynthParameterPack
{
    enum FreqMode : bool
    {
        Fixed = false,
        Relative = true
    };
    std::bitset<32> freqMode;
    Envelope<32> perOperatorFreq;

    Envelope<1024> modulationMatrix;
    Envelope<32> operatorVolume;
};


class FMSynth
{
public:
    FMSynth(size_t sampleRate, size_t samplesPerFrame, size_t channels);
    ~FMSynth();

    void setSampleRate(size_t newSampleRate);
    void setSamplesPerFrame(size_t newSamplesPerFrame);
    void setNumChannels(size_t newNumChannels);

    [[nodiscard]] size_t notePress(float freq);
    void noteRelease(size_t slot);
    void noteKill(size_t slot);
    void recycleNoteSlots();
    void renderAudio();
    void getRenderedAudio(float* outputBuffer);

    SynthParameterPack parameterEnvelopes;
    std::mutex parameterMutex;

    //debug functions
    void allInUse();

private:
    void renderAudioDebug();

    size_t currentCycle = 0;
    size_t sampleRate;
    size_t nChannels;
    size_t samplesPerFrame;
    float invSampleRate;
    float* gpuOperatorBuffer;
    float* gpuAudioBuffer;

    size_t noteStartCycle[FMVoices];

    GPUOperatorPack<32>* gpuOps;
    size_t allocPitch;

    std::array<float, FMVoices> notesBaseFrequency;

    std::bitset<FMVoices> inUse;
    std::bitset<FMVoices> released;
    std::array<float, 32> releasedFrequencies[FMVoices];
    std::array<float, 1024> releasedModMatrices[FMVoices];
    std::array<float, 32> releasedVolumes[FMVoices];

    //void rescaleParameters(float oldSampleRate, float newSampleRate);
    //std::vector<float> audioDump;
};



void testFMSynthesis(size_t samplesPerBlock);
void testFMSynthesis_CPU(size_t samplesPerBlock);