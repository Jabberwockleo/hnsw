/**
 * File              : ann.h
 * Author            : Wan Li
 * Date              : 05.09.2019
 * Last Modified Date: 05.09.2019
 * Last Modified By  : Wan Li
 */

#ifndef ANN_H
#define ANN_H

#include <string>
#include <iostream>
#include "hnswlib/hnswlib.h"

class ANN {
    public:
        ANN(const std::string &metric_space_name, const size_t feature_dim) :
            _metric_space_name(metric_space_name),
            _feature_dim(feature_dim) {
            _normalize = false;
            if (metric_space_name == "l2") {
                _metric_space = new hnswlib::L2Space(feature_dim);
            } else if (metric_space_name == "inner-product") {
                _metric_space = new hnswlib::InnerProductSpace(feature_dim);
            } else if (metric_space_name == "cosine") {
                _metric_space = new hnswlib::InnerProductSpace(feature_dim);
                _normalize = true;
            }
            _graph = nullptr;
            _num_threads_default = 1;
            _is_ep_added = true;
            _is_index_inited = false;
        }
    private:
        std::string _metric_space_name;
        size_t _feature_dim;
        hnswlib::SpaceInterface<float> *_metric_space;
        hnswlib::HierarchicalNSW<float> *_graph;

        int _num_threads_default;
        bool _normalize;
        bool _is_ep_added;
        bool _is_index_inited;
};

#endif
