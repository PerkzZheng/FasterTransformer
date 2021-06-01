/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <thread>
#include "fastertransformer/triton_backend/gpt_triton_backend.hpp"

using namespace fastertransformer;

int thread_main(int argc, char* argv[],
                int node_id, int device_id, int world_rank, int world_size,
                std::shared_ptr<AbstractTransformerModel> model,
                std::vector<ncclUniqueId> nccl_ids,
                ncclComm_t tensor_para_nccl_comm,
                ncclComm_t layer_para_nccl_comm)
{
    CUDACHECK(cudaSetDevice(device_id));
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device_id));
    printf("Device %s node_id = %d, device_id = %d, world_size = %d\n", prop.name, node_id, device_id, world_size);

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));


    auto modelInstance = model->createModelInstance(node_id, device_id, world_size, stream);
    auto param_instance = model->createParamInstance(node_id, device_id, world_rank, world_size, stream, nccl_ids);
    //param_instance->init_nccl_from_ids(nccl_ids);
    param_instance->init_nccl_from_comms(tensor_para_nccl_comm, layer_para_nccl_comm);
    modelInstance->set_param(param_instance.get());
    printf("model instance is created \n");

    std::string ini_name;
    if(argc >= 2)
        ini_name = std::string(argv[1]);
    else
        ini_name = "../sample/cpp/gpt_config.ini";
    auto request = prepareRequest(ini_name, "./start_ids.csv");
    if(node_id == 0 && device_id == 0)
        check_inputs(request);

    printf("request is created \n");

    size_t free_bytes, total_bytes;
    check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
    float free = (float)(free_bytes) / 1024.0 / 1024.0 / 1024.0;
    float total = (float)(total_bytes) / 1024.0 / 1024.0 / 1024.0;
    printf("after allocation, free %.2f GB total %.2f GB\n", free, total);

    auto output = modelInstance->forward(request);

    if(node_id == 0 && device_id == 0)
        check_outputs(output);

    printf("Thread node id = %d, devide id = %d ends\n", node_id, device_id);
    return 0;
}

int main(int argc, char *argv[])
{
    std::string ini_name;
    if(argc >= 2)
        ini_name = std::string(argv[1]);
    else
        ini_name = "../sample/cpp/gpt_config.ini";
    // int node_id = argc >= 5 ? std::stoi(argv[2]) : 0;
    // int gpu_size = argc >= 5 ? std::stoi(argv[3]) : 1;
    // int world_size = argc >= 5 ? std::stoi(argv[4]) : 1;
    int node_id, gpu_size, world_size;
    /* we assume that each node has the same number of GPUs
    each rank points to one layer parallel unit
    for example:
    Node 0: GPU 0, GPU 1,GPU 2, GPU 3,GPU 4, GPU 5,GPU 6, GPU 7
    Node 1: GPU 0, GPU 1,GPU 2, GPU 3,GPU 4, GPU 5,GPU 6, GPU 7
    Layer Paralel Size = 2
    Tensor Parallel Size = 8
    Then MPI will issue two ranks (rank 0 handles Node0, rank1 handles Node1)
    */
    CUDACHECK(cudaGetDeviceCount(&gpu_size));

    MPICHECK( MPI_Init(&argc, &argv));

    int layer_para_rank, layer_para_mpi_size;
    MPICHECK( MPI_Comm_rank(MPI_COMM_WORLD, &layer_para_rank));
    MPICHECK( MPI_Comm_size(MPI_COMM_WORLD, &layer_para_mpi_size));
    

    auto model = AbstractTransformerModel::createGptModel(ini_name);
    int tensor_para_size = model->get_tensor_para_size();
    int layer_para_size =  model->get_layer_para_size();

    world_size = tensor_para_size * layer_para_size;

    //CHECK if number of MPI Ranks == Layer parallel size and check if number of GPUs is enough
    if( layer_para_mpi_size != layer_para_size) MPI_Abort(MPI_COMM_WORLD, -1);

    node_id = (layer_para_rank * tensor_para_size) / gpu_size;

    std::vector<ncclUniqueId> nccl_ids;

    if(layer_para_rank == 0)
    {
        nccl_ids = model->create_nccl_ids(world_size);
    }
    int nccl_size = nccl_ids.size();
    // MPI_Barrier(MPI_COMM_WORLD);
    MPICHECK(MPI_Bcast(&nccl_size, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if(layer_para_rank != 0) nccl_ids = std::vector<ncclUniqueId>(nccl_size);
    // MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < nccl_ids.size(); i++)
    {
        MPICHECK( MPI_Bcast(&nccl_ids[i], sizeof(nccl_ids[i]), MPI_BYTE, 0, MPI_COMM_WORLD));
    }

    printf("NCCL IDs size %ld\n",nccl_ids.size());

    //nccl initialization inside one MPI rank
    //each MPI Rank handles all tensor parallel units inside one layer parallel unit
    ncclComm_t tensor_nccl_comms[tensor_para_size];
    ncclComm_t layer_nccl_comms[tensor_para_size];

    NCCLCHECK(ncclGroupStart());
    for (int tensor_para_rank = 0; tensor_para_rank < tensor_para_size; tensor_para_rank ++) {

        int gid = tensor_para_rank + layer_para_rank * tensor_para_size;

        ncclUniqueId tensor_para_nccl_uid = nccl_ids[layer_para_rank];
        ncclUniqueId layer_para_nccl_uid  = nccl_ids[layer_para_size + tensor_para_rank];

        CUDACHECK(cudaSetDevice(gid % gpu_size));
        NCCLCHECK( ncclCommInitRank(&tensor_nccl_comms[tensor_para_rank], tensor_para_size, tensor_para_nccl_uid, tensor_para_rank));
        NCCLCHECK( ncclCommInitRank(&layer_nccl_comms[tensor_para_rank], layer_para_size, layer_para_nccl_uid, layer_para_rank));

    }
    NCCLCHECK(ncclGroupEnd());

    // MPI_Barrier(MPI_COMM_WORLD);
    std::vector<std::thread> threads;
    for(int tensor_para_rank = 0; tensor_para_rank < tensor_para_size; tensor_para_rank ++) {
        // printf("tensor_para_size %d gid : %d\n",tensor_para_size, gid);
        int world_rank = tensor_para_rank + layer_para_rank * tensor_para_size;
        int gid = world_rank % gpu_size;
        threads.push_back(std::thread(thread_main, argc, argv,
                                        node_id, gid, world_rank, world_size,
                                        model, nccl_ids,
                                        tensor_nccl_comms[tensor_para_rank],
                                        layer_nccl_comms[tensor_para_rank]));
    }

    for(auto & t : threads) {
        t.join();
    }

    MPICHECK(MPI_Finalize());

    printf("ALL THREADs END\n");
}
