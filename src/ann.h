/**
 * File              : ann.h
 * Author            : Wan Li
 * Date              : 05.09.2019
 * Last Modified Date: 09.09.2019
 * Last Modified By  : Wan Li
 */

#pragma once

#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
#include "hnswlib/hnswlib.h"

namespace ann {

struct QueryResult {
    size_t topk;
    std::vector<hnswlib::labeltype> indices;
    std::vector<float> distances;
    explicit QueryResult() {
        this->topk = 0;
        this->indices.clear();
        this->distances.clear();
    }

    explicit QueryResult(std::priority_queue<std::pair<float, hnswlib::labeltype> > knn) {
        this->topk = knn.size();
        this->indices = std::move(std::vector<hnswlib::labeltype>(this->topk));
        this->distances = std::move(std::vector<float>(this->topk));
        for (int i = this->topk - 1; i >= 0; --i) {
            auto dist_index_pair = knn.top();
            this->distances[i] = dist_index_pair.first;
            this->indices[i] = dist_index_pair.second;
            knn.pop();
        }
    }
    
    std::string dump() {
        std::stringstream ss;
        ss << "[";

        ss << "topk:" << this->topk << ";";

        ss << "indices:";
        for (int i = 0; i < this->topk; ++i) {
            ss << this->indices[i];
            if (i != this->topk - 1) {
                ss << ",";
            }
        }
        ss << ";";

        ss << "distances:";
        for (int i = 0; i < this->topk; ++i) {
            ss << this->distances[i];
            if (i != this->topk - 1) {
                ss << ",";
            }
        }
        
        ss << "]";
        return ss.str();
    }
};

class ANN {
    public:
        explicit ANN(const std::string &metric_space_name,
            const size_t feature_dim, int num_threads=1);
        ANN() = delete;
        ~ANN() {
            delete _metric_space;
            delete _graph;
        }

        // Create a new HNSW graph
        //     max_nodes: upper bound for capacity
        //     M: lower bound for degree
        //     ef_construction: search space during construction
        int create_index(const size_t max_nodes, const size_t M=16,
            const size_t ef_construction=200, const size_t random_seed=100);

        // Batch insert nodes to graph
        //     data: array of feature vectors
        //     data_indices: indices (intrinsic labels) for each data point
        //         optional, if an empty vector is passed, labels will be
        //         inferred internally.
        int insert_nodes(const std::vector<std::vector<float> > &data,
            const std::vector<size_t> &data_indices);

        // Persist index graph
        int save_index(const std::string &path);

        // Load index graph from disk
        int load_index(const std::string &path, const size_t max_nodes);

        // Batch query approximate nearest neighbors to points in feature space
        std::vector<QueryResult> knn_query(
            const std::vector<std::vector<float> > &queries, size_t k);

    private:
        void normalize_vector(const float *data, float *norm_data);

    private:
        std::string _metric_space_name;
        size_t _feature_dim;
        hnswlib::SpaceInterface<float> *_metric_space;
        hnswlib::HierarchicalNSW<float> *_graph;

        int _num_threads;
        bool _normalize;
        bool _is_index_initialized;
        bool _is_entry_point_added; // safe guard for parallel building

        hnswlib::labeltype _current_node_label;
};

} // namespace ann
