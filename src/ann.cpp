/**
 * File              : ann.cpp
 * Author            : Wan Li
 * Date              : 06.08.2019
 * Last Modified Date: 06.09.2019
 * Last Modified By  : Wan Li
 */

#include "ann.h"

#include <atomic>
#include <thread>

namespace ann {

template<class Function>
inline void parallel_for_loop(size_t start, size_t end,
    size_t num_threads, Function fn) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }

    if (num_threads == 1) {
        for (size_t id = start; id < end; ++id) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        std::exception_ptr last_exception = nullptr;
        std::mutex last_exception_mutex;

        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            threads.push_back(std::thread([&, thread_id] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if ((id >= end)) {
                        break;
                    }

                    try {
                        fn(id, thread_id);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(last_exception_mutex);
                        last_exception = std::current_exception();
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (last_exception) {
            std::rethrow_exception(last_exception);
        }
    }
}

ANN::ANN(const std::string &metric_space_name, const size_t feature_dim,
    int num_threads) :
    _metric_space_name(metric_space_name),
    _feature_dim(feature_dim),
    _num_threads(num_threads) {
    this->_normalize = false;
    if (metric_space_name == "l2") {
        this->_metric_space = new hnswlib::L2Space(feature_dim);
    } else if (metric_space_name == "inner-product") {
        this->_metric_space = new hnswlib::InnerProductSpace(feature_dim);
    } else if (metric_space_name == "cosine") {
        this->_metric_space = new hnswlib::InnerProductSpace(feature_dim);
        this->_normalize = true;
    }
    this->_graph = nullptr;
    this->_is_entry_point_added = true;
    this->_is_index_initialized = false;
}

int ANN::create_index(const size_t max_nodes, const size_t M,
    const size_t ef_construction, const size_t random_seed) {
    if (this->_graph != nullptr) {
        // Already initialized
        return -1;
    }
    this->_current_node_label = 0;
    this->_graph = new hnswlib::HierarchicalNSW<float>(
        this->_metric_space, max_nodes,
        M, ef_construction, random_seed);
    this->_is_index_initialized = true;
    this->_is_entry_point_added = false;
    return 0;
}

int ANN::insert_nodes(const std::vector<std::vector<float> > &data,
    const std::vector<size_t> &data_indices) {
    size_t num_entries = data.size();
    if (!data_indices.empty() && (data_indices.size() != data.size())) {
        // Bad index parameter
        return -1;
    }

    int num_threads = this->_num_threads;
    if (num_entries < num_threads*4) {
        num_threads = 1;
    }

    int start = 0;
    if (!this->_is_entry_point_added) {
        // safe guard for parallel building
        size_t idx = data_indices.size() ?
            data_indices[0] : this->_current_node_label;
        if (this->_normalize) {
            std::vector<float> normvec(this->_feature_dim);
            normalize_vector(data.at(0).data(), normvec.data());
            this->_graph->addPoint((void *)normvec.data(), idx);
        } else {
            this->_graph->addPoint((void *)(data.at(0).data()), idx);
        }
        start = 1;
        this->_is_entry_point_added = true;
    }
    
    std::vector<float> parallel_normvec(num_threads * this->_feature_dim);
    parallel_for_loop(start, num_entries, num_threads,
        [&](size_t internal_idx, size_t thread_id) {
            size_t idx = data_indices.size() ?
                data_indices[0] : (this->_current_node_label + internal_idx);
            if (this->_normalize) {
                size_t parallel_normvec_idx0_pos = thread_id * this->_feature_dim;
                normalize_vector(data.at(internal_idx).data(),
                    (parallel_normvec.data() + parallel_normvec_idx0_pos));
                this->_graph->addPoint(
                    (void *)(parallel_normvec.data() + internal_idx),
                    idx);
            } else {
                this->_graph->addPoint((void *)data.at(internal_idx).data(), idx);
            }
        });
    this->_current_node_label += num_entries;

    return 0;
}

int ANN::save_index(const std::string &path) {
    this->_graph->saveIndex(path);
    return 0;
}

int ANN::load_index(const std::string &path, const size_t max_nodes) {
    if (this->_graph != nullptr) {
        delete this->_graph;
    }
    this->_graph = new hnswlib::HierarchicalNSW<float>(
        this->_metric_space,
        path,
        false,
        max_nodes);
    this->_current_node_label = this->_graph->cur_element_count;
    return 0;
}

std::vector<QueryResult> ANN::knn_query(
            const std::vector<std::vector<float> > &queries, size_t k) {
    return std::vector<QueryResult>();
}

void ANN::normalize_vector(const float *data, float *norm_data) {
    float norm = 0.0f;
    for(int i = 0; i < this->_feature_dim; ++i) {
        norm += data[i] * data[i];
    }
    norm = 1.0f / (sqrtf(norm) + 1e-30f);
    for (int i = 0; i < this->_feature_dim; ++i) {
        norm_data[i] = data[i] * norm;
    }
}

} // namespace ann