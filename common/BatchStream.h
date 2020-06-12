/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include "NvInfer.h"
#include "common.h"
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <vector>

using namespace std;
class IBatchStream
{
public:
    virtual void reset(int firstBatch) = 0;
    virtual bool next() = 0;
    virtual void skip(int skipCount) = 0;
    virtual float* getBatch() = 0;
    virtual float* getLabels() = 0;
    virtual int getBatchesRead() const = 0;
    virtual int getBatchSize() const = 0;
    virtual nvinfer1::Dims getDims() const = 0;
};



//********************************************************************************************************//

class BINcifarBatchStream 
{
public:
    BINcifarBatchStream(int batchSize, int maxBatches,nvinfer1::Dims Dims)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{Dims}  
    {
        readDataFile();
    //    readLabelsFile();
    }

    void reset(int firstBatch) 
    {
        mBatchCount = firstBatch;
    }

    bool next() 
    {
        if (mBatchCount == mMaxBatches-1 )
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) 
    {
        mBatchCount += skipCount;
    }

    float* getBatch() 
    {   
        count++;
 
        int num = datasetnum/mBatchSize;//drop last batch
        int index = count % num;

        return mData.data() + (index * mBatchSize * samplesCommon::volume(mDims));

    }

    char* getLabel() 
    {
      
        return mLabels.data() + (count * mBatchSize);
    }

    int getBatchesRead() const 
    {
        return mBatchCount;
    }

    int getBatchSize() const 
    {
        return mBatchSize;
    }
    nvinfer1::Dims getDims() const 
    {
        return mDims;
    }
    nvinfer1::Dims getImageDims() 
    {
        return mDims;
    }
    int count{-1};
    int datasetnum{10000};
private:
    void readDataFile()
    {
    int channels =3;
    int height =32;
    int width =32;
    
    vector<uint8_t> fileData(datasetnum*channels * height * width);
   
    mData.resize(datasetnum*channels * height * width);
    mLabels.resize(datasetnum);
    uint8_t *p_data;
    p_data = fileData.data();
    char *l_data;
    l_data = mLabels.data();

    cifar10 normalize;

    ifstream fin;
    fin.open("/home/peppa1/test_batch.bin",ifstream::binary);
    assert(fin && "Attempting to read from a file that is not open.");
    
    for(int i = 0;i< datasetnum; i++)
    {

       fin.read(reinterpret_cast<char*>(l_data),1);

       fin.read(reinterpret_cast<char*>(p_data),channels * height * width); 

       l_data+=1;
       p_data += width * height * channels;

      for (int c = 0; c < channels; ++c)
     {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = i * channels *height *width + ( c * height * width + h * width + w );

                mData[dstIdx] = (float) (fileData[dstIdx] / normalize.scale -normalize.mean.at(c)) / normalize.std.at(c);
            }
        }
     }
    }
fin.close();
  //  fclose(fpr);
    gLogError << "success res  "  << std::endl;
    }

    void readLabelsFile()
    {

    }

    int mBatchSize{0};
    int mBatchCount{-1}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    Dims mDims{};
    std::vector<float> mData{};
    std::vector<char> mLabels{};
};
//********************************************************************************************************//

