/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *d
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THEs
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "sampleOptions.h"
#include "sampleEngines.h"
#include "BatchStream.h"

using namespace nvinfer1;
using namespace sample;
using namespace std;


int picwidth = 32;
int picheight = 32;
int numclass = 10;
int success = 0;
int iteration = 1500;

float percentile(float percentage, std::vector<float>& times)
{
    int all = static_cast<int>(times.size());
    int exclude = static_cast<int>((1 - percentage / 100) * all);
    if (0 <= exclude && exclude <= all)
    {
        std::sort(times.begin(), times.end());
        return times[all == exclude ? 0 : all - 1 - exclude];
    }
    return std::numeric_limits<float>::infinity();
}

bool verifyOutput(float* prob, int count, char* truth, int batchsize) //verify top 5 correction
{
     
    vector<float> output(prob, prob + batchsize*numclass);

    for(int j = 0; j < batchsize; j++){

    auto ind = samplesCommon::argsort(output.begin()+j* numclass ,output.begin()+(j+1)* numclass,true);

  //  gLogInfo << " ind[0]:" << std::to_string(ind[0])<<" Truth "<<std::to_string(truth[j])<<std::endl;
 /*     for (int i = 1; i <= 5; ++i)
      {
        if (std::to_string(ind[i-1]).compare(groundtruth[j+count*mParams.batchSize]) == 0 ) { success +=1;}
      }
*/
    if (std::to_string(ind[0]).compare(std::to_string(truth[j])) == 0 ) { success +=1;}
    }



 //   std::cout<<"Top 1: "<<(float)success/mParams.batchSize<<std::endl;
    return true;
}
bool doInference(ICudaEngine& engine, const InferenceOptions& inference, const AllOptions& options)
{
   IExecutionContext* context = engine.createExecutionContext();

    // Dump inferencing time per layer basis
    SimpleProfiler profiler("Layer time");
    if (options.reporting.profile)
    {
        context->setProfiler(&profiler);
    }

    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (!engine.bindingIsInput(b))
        {
            continue;
        }
        auto dims = context->getBindingDimensions(b);
        if (dims.d[0] == -1)
        {
            auto shape = inference.shapes.find(engine.getBindingName(b));
            if (shape == inference.shapes.end())
            {
                gLogError << "Missing dynamic batch size in inference" << std::endl;
                return false;
            }
            dims.d[0] = shape->second.d[0];
            context->setBindingDimensions(b, dims);
        }
    }

  
        bufferManager.copyOutputToHost();
        int nbBindings = engine.getNbBindings();
        for (int i = 0; i < nbBindings; i++)
        {
            if (!engine.bindingIsInput(i))
            {
                const char* tensorName = engine.getBindingName(i);
                gLogInfo << "Dumping output tensor " << tensorName << ":" << std::endl;
                bufferManager.dumpBuffer(gLogInfo, tensorName);
            }
        }
    }


    void* buffers[2];

    float* prob = new float[inference.batch * numclass];
    int count=0;

    float totalTime{0.0f};
    float ms{0.0f},max{0.0f},min{1000.0f};

    float h2dtime{0.0f},d2htime{0.0f}; 
    
    string precision = options.build.int8 ? "INT8":(options.build.fp16 ? "FP16" : "FP32");
    string device = (options.system.DLACore == -1 )? "GPU" :((options.system.DLACore == 0 )? "DLA0" : "DLA1");
    string name0 = "/home/peppa2/resultsdla/" + options.build.network + precision + "h2dtime_batch-" +device+ ".txt";
    string name1 = "/home/peppa2/resultsdla/" + options.build.network + precision + "comptime_batch-"+device + ".txt";
    string name2 = "/home/peppa2/resultsdla/" + options.build.network + precision + "d2htime_batch-" +device + ".txt";

    ofstream outputfile0( name0 ,ios::app);
    ofstream outputfile1( name1 ,ios::app);
    ofstream outputfile2( name2 ,ios::app);

    outputfile0<< "      batchsize is     "<<inference.batch<<std::endl;
    outputfile1<< "      batchsize is     "<<inference.batch<<std::endl;
    outputfile2<< "      batchsize is     "<<inference.batch<<std::endl;
    
    CHECK(cudaMalloc((void**)&buffers[0], inference.batch*3*picwidth*picheight*sizeof(float)));
    CHECK(cudaMalloc((void**)&buffers[1], inference.batch*numclass*sizeof(float)));
    
    if (!context)
    {
        return false;
    }

    nvinfer1::Dims InputDims{3,3,picwidth,picheight};//picture size Another '3' means this is a 3 DimsChannelsHeightWeoghtss
  //  auto InputDims = context->getBindingDimensions(0);
    BINcifarBatchStream pStream( inference.batch,iteration,InputDims);


    while (pStream.next())// Read the input data into the managed buffers
    {  
   
	cudaEvent_t start0, end0;
        CHECK(cudaEventCreate(&start0));
        CHECK(cudaEventCreate(&end0));
        cudaEventRecord(start0, 0);
      
     //  int buffsize = mParams.int8 ? mParams.batchSize*3*picwidth*picheight*sizeof(float):mParams.batchSize*3*picwidth*picheight*sizeof(float)/2;
        int buffsize = inference.batch*3*picwidth*picheight*sizeof(float);  

        CHECK(cudaMemcpy(buffers[0], pStream.getBatch(), buffsize, cudaMemcpyHostToDevice));
        // Memcpy from host input buffers to device input buffersss

	cudaEventRecord(end0, 0);
        cudaEventSynchronize(end0);
        cudaEventElapsedTime(&h2dtime, start0, end0);
        outputfile0<< "batch "<<count<<" h2d time :"<<h2dtime<<" ms"<<std::endl;
        cudaEventDestroy(start0);
        cudaEventDestroy(end0);
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // Use CUDA events to measure inference time
        cudaEvent_t start1, end1;
        CHECK(cudaEventCreateWithFlags(&start1,cudaEventDefault));
        CHECK(cudaEventCreateWithFlags(&end1,cudaEventDefault));
        cudaEventRecord(start1, stream);

        bool status = context->enqueue(inference.batch, buffers, stream, nullptr);
        if (!status)
        {
            return false;
        }
        cudaEventRecord(end1, stream);
        cudaEventSynchronize(end1);
        cudaEventElapsedTime(&ms, start1, end1);

    
        // Memcpy from device output buffers to host output buffers
        totalTime += ms;
	
	if(min > ms) min =ms;
	if(max < ms) max =ms;
        if(count!=0 && ( count + 1 ) % 100 == 0) 
        {
	gLogInfo << "Processing 100 batch avg time is"<<(float)totalTime/100<<" ms!" << std::endl;
	gLogInfo << "Processing 100 batch min time is"<<min<<" ms!" << std::endl;
	gLogInfo << "Processing 100 batch max time is"<<max<<" ms!" << std::endl;
	totalTime = 0;
	}  
      
	outputfile1<< "batch "<<count<<" computation time :"<<ms<<" ms"<<std::endl;
        cudaEventDestroy(start1);
        cudaEventDestroy(end1);
        CHECK(cudaStreamDestroy(stream));

	cudaEvent_t start2, end2;
        CHECK(cudaEventCreate(&start2));
        CHECK(cudaEventCreate(&end2));
        cudaEventRecord(start2, 0);

        CHECK(cudaMemcpy(prob,buffers[1],inference.batch*10*sizeof(float), cudaMemcpyDeviceToHost));
	
	cudaEventRecord(end2, 0);
        cudaEventSynchronize(end2);
        cudaEventElapsedTime(&d2htime, start2, end2);
        outputfile2<< "batch "<<count<<" d2h time :"<<d2htime<<" ms"<<std::endl;
        cudaEventDestroy(start2);
        cudaEventDestroy(end2);

 
        if(!verifyOutput(prob,count,pStream.getLabel(),inference.batch))
        {
         gLogInfo << "Processing batches "<<count<<" wrong!" << std::endl;
        }
        count++;
    }

    gLogInfo << "success is "<< success << std::endl;
    outputfile0.close();
    outputfile1.close();
    outputfile2.close();
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));

    if (options.reporting.profile)
    {
        gLogInfo << profiler;
    }


    context->destroy();

    return true;
}

int main(int argc, char** argv)
{
    const std::string sampleName = "TensorRT.trtexec";
    auto sampleTest = gLogger.defineTest(sampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    Arguments args = argsToArgumentsMap(argc, argv);
    AllOptions options;

    if (!args.empty())
    {
        bool failed{false};
        try
        {
            options.parse(args);

            if (!args.empty())
            {
                for (const auto& arg : args)
                {
                    gLogError << "Unknown option: " << arg.first << " " << arg.second << std::endl;
                }
                failed = true;
            }
        }
        catch (const std::invalid_argument& arg)
        {
            gLogError << arg.what() << std::endl;
            failed = true;
        }

        if (failed)
        {
            AllOptions::help(std::cout);
            std::cout << "Note: the following options are not fully supported in trtexec:"
                         " dynamic shapes, multistream/threads, cuda graphs, json logs,"
                         " and actual data IO" << std::endl;
            return gLogger.reportFail(sampleTest);
        }
    }
    else
    {
        options.helps = true;
    }

    if (options.helps)
    {
        AllOptions::help(std::cout);
        std::cout << "Note: the following options are not fully supported in trtexec:s"
                     " dynamic shapes, multistream/threads, cuda graphs, json logs,"
                     " and actual data IO" << std::endl;
        return gLogger.reportPass(sampleTest);
    }

    gLogInfo << options;
    if (options.reporting.verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }

    cudaSetDevice(options.system.device);

    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    for (const auto& pluginPath : options.system.plugins)
    {
        gLogInfo << "Loading supplied plugin library: " << pluginPath << std::endl;
        samplesCommon::loadLibrary(pluginPath);
    }

    ICudaEngine* engine{nullptr};
    if (options.build.load)
    {
        engine = loadEngine(options.build.engine, options.system.DLACore, gLogError);
    }
    else
    {
        engine = modelToEngine(options.model, options.build, options.system, gLogError);
    }
    if (!engine)
    {
        gLogError << "Engine could not be created" << std::endl;
        return gLogger.reportFail(sampleTest);
    }
    if (options.build.save)
    {
        saveEngine(*engine, options.build.engine, gLogError);
    }

    if (!options.inference.skip)
    {
        if (options.build.safe && options.system.DLACore >= 0)
        {
            gLogInfo << "Safe DLA capability is detected. Please save DLA loadable with --saveEngine option, "
                        "then use dla_safety_runtime to run inference with saved DLA loadable, "
                        "or alternatively run with your own application" << std::endl;
            return gLogger.reportFail(sampleTest);
        }
        if (!doInference(*engine, options.inference, options))
        {
            gLogError << "Inference failure" << std::endl;
            return gLogger.reportFail(sampleTest);
        }
    }
    engine->destroy();

    return gLogger.reportPass(sampleTest);
}
