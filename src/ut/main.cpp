/**
 * File              : main.cpp
 * Author            : Wan Li
 * Date              : 05.09.2019
 * Last Modified Date: 05.09.2019
 * Last Modified By  : Wan Li
 */

#include <iostream>
#include <fstream>
#include "../ann.h"
#include "../dict.h"

template <typename T>
int print_embedding(const std::vector<T> &vec) {
    for (auto &e : vec) {
        std::cout << e << ", ";
    }
    return 0;
}

int load_titles(const char *fn, std::map<std::string, std::string> *w2t) {
    std::string fn_embedding = std::string(fn);
    std::string line;
    std::ifstream infile(fn_embedding);
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<std::string> rr((std::istream_iterator<std::string>(iss)),
            std::istream_iterator<std::string>());
        (*w2t)[rr[0]] = rr[1];
    }
    return 0;
}

int main(int argc, char **argv) {

    ann::Dict dict;
    {
        int err_code = 0;
        err_code = dict.load_dict("../../data/");
        if (err_code != 0) {
            std::cout << "ERR: " << err_code << std::endl;
            return -1;
        }
        // TEST: Dict
        std::cout << "== TEST Dict ==" << std::endl;
        std::string q("41f69ee711960cd5ac764721d7b3733b");
        std::cout << "query: " << q << std::endl;
        size_t idx;
        if (dict.index_for_word(q, &idx) == 0) {
            std::cout << "index: " << idx << std::endl;
        } else {
            std::cout << "index: " << "ERR" << std::endl;
        }
    
        std::vector<float> emb;
        if (dict.embedding_for_word(q, &emb) == 0) {
            std::cout << "embedding: ";
            print_embedding<float>(emb);
            std::cout << std::endl;
        } else {
            std::cout << "embedding: " << "ERR" << std::endl;
        }

        std::string word;
        if (dict.word_for_index(0, &word) == 0) {
            std::cout << "word: " << word << std::endl;
        } else {
            std::cout << "word: " << "ERR" << std::endl;
        }
    }

    {
        ann::ANN ann("cosine", 64);
        // TEST: build graph
        std::cout << "== TEST ANN ==" << std::endl;
        std::cout << "Creating graph.." << std::endl;
        ann.create_index(1023);
        std::cout << "Building graph.." << std::endl;
        ann.insert_nodes(dict.embedding_matrix(), std::vector<size_t>());
        std::cout << "Saving graph.." << std::endl;
        ann.save_index("./graph.bin");
        std::cout << "Loading graph.." << std::endl;
        ann::ANN ann2("cosine", 64);
        ann2.load_index("./graph.bin", 1023);
        std::cout << "KNN query.." << std::endl;
        std::vector<std::vector<float> > batch_q;
        std::vector<float> emb;
        std::string word("25486855a059698552b3d213df29e2c2");
        dict.embedding_for_word(word, &emb);
        batch_q.emplace_back(emb);
        auto arr_qs = ann2.knn_query(batch_q, 10);
        for (auto &qs : arr_qs) {
            std::cout << qs.dump() << std::endl;
        }
        std::map<std::string, std::string> w2t;
        load_titles("../../data/item_title.txt", &w2t);
        std::cout << "IN:  " << w2t[word] << std::endl;
        for (auto idx : arr_qs[0].indices) {
            std::string word;
            dict.word_for_index(idx, &word);
            std::cout << w2t[word] << std::endl;
        }
    }
    std::cout << "DONE" << std::endl;
    return 0;
}
