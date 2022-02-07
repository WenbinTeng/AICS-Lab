#include "inference.h"
#include "cnrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stdlib.h"
#include <sys/time.h>
#include <time.h>

namespace StyleTransfer {

Inference :: Inference(std::string offline_model) {
    offline_model_ = offline_model;
}

void Inference :: run(DataTransfer* DataT) {
    cnrtInit(0);
    cnrtModel_t model;
    cnrtLoadModel(&model, offline_model_.c_str());

    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, 0);
    cnrtSetCurrentDevice(dev);

    cnrtFunction_t function;
    cnrtCreateFunction(&function);
    cnrtExtractFunction(&function, model, "subnet0");
    int inputNum, outputNum;
    int64_t *inputSizeS, *outputSizeS;
    cnrtGetInputDataSize(&inputSizeS, &inputNum, function); 
    cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);

    void **inputCpuPtrS = (void **)malloc(inputNum * sizeof(void *));
    void **outputCpuPtrS = (void **)malloc(outputNum * sizeof(void *));

    void **inputMluPtrS = (void **)malloc(inputNum * sizeof(void *));
    void **outputMluPtrS = (void **)malloc(outputNum * sizeof(void *));
    void **inputHalf = (void **)malloc(inputNum * sizeof(void *));
    void **outputHalf = (void **)malloc(outputNum * sizeof(void *));

    for(int i = 0; i < inputNum; i++) {
        inputCpuPtrS[i] = malloc(inputSizeS[i] * 2);
        inputHalf[i] = malloc(inputSizeS[i]);
        cnrtMalloc(&(inputMluPtrS[i]), inputSizeS[i]);
        int dimValues[] = {1, 3, 256, 256};
        int dimOrder[] = {0, 2, 3, 1};
        CNRT_CHECK(cnrtTransDataOrder(DataT->input_data, CNRT_FLOAT32, inputCpuPtrS[i], 4, dimValues, dimOrder));
        CNRT_CHECK(cnrtCastDataType(inputCpuPtrS[i], CNRT_FLOAT32, inputHalf[i], CNRT_FLOAT16, inputSizeS[i]/2, NULL));
        cnrtMemcpy(inputMluPtrS[i], inputHalf[i], inputSizeS[i], CNRT_MEM_TRANS_DIR_HOST2DEV);        
    }

    for (int i = 0; i < outputNum; i++) {
        outputCpuPtrS[i] = malloc(outputSizeS[i] * 2);
        outputHalf[i] = malloc(outputSizeS[i]);
        cnrtMalloc(&(outputMluPtrS[i]), outputSizeS[i]);
    }

    void **param = (void **)malloc(sizeof(void *) * (inputNum + outputNum));
    for (int i = 0; i < inputNum; ++i) {
        param[i] = inputMluPtrS[i];
    }
    for (int i = 0; i < outputNum; ++i) {
        param[inputNum + i] = outputMluPtrS[i];
    }

    cnrtRuntimeContext_t ctx;
    CNRT_CHECK(cnrtCreateRuntimeContext(&ctx, function, NULL));

    CNRT_CHECK(cnrtSetRuntimeContextDeviceId(ctx, 0));
    CNRT_CHECK(cnrtInitRuntimeContext(ctx, NULL));

    cnrtQueue_t queue;
    CNRT_CHECK(cnrtRuntimeContextCreateQueue(ctx, &queue));

    CNRT_CHECK(cnrtInvokeRuntimeContext(ctx, param, queue, NULL));
    
    CNRT_CHECK(cnrtSyncQueue(queue));

    for(int i = 0; i < outputNum; i++){
        CNRT_CHECK(cnrtMemcpy(outputHalf[i], outputMluPtrS[i], outputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2HOST));
    }
    CNRT_CHECK(cnrtCastDataType(outputHalf[0], CNRT_FLOAT16, outputCpuPtrS[0], CNRT_FLOAT32, outputSizeS[0]/2, NULL));
    int dimValues[] = {1, 256, 256, 3};
    int dimOrder[] = {0, 3, 1, 2};
    DataT->output_data = (float *)malloc(outputSizeS[0]*2);
    CNRT_CHECK(cnrtTransDataOrder(outputCpuPtrS[0], CNRT_FLOAT32, DataT->output_data, 4, dimValues, dimOrder));

    for(int i = 0; i < inputNum; i++){
        free(inputCpuPtrS[i]);
        free(inputHalf[i]);
        cnrtFree(inputMluPtrS[i]);
    }
    for (int i = 0; i < outputNum; i++) {
        free(outputCpuPtrS[i]);
        free(outputHalf[i]);
        cnrtFree(outputMluPtrS[i]);
    }
    free(inputCpuPtrS);
    free(inputHalf);
    free(outputCpuPtrS);
    free(outputHalf);
    free(inputMluPtrS);
    free(outputMluPtrS);
    free(param);

    cnrtDestroyQueue(queue);
    cnrtDestroyRuntimeContext(ctx);
    cnrtDestroyFunction(function);
    cnrtUnloadModel(model);
    cnrtDestroy();
}

} // namespace StyleTransfer
