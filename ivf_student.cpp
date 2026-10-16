#include "ivf_search.h"

#include <stdexcept>

namespace ivf {

void search_std_thread(const IVFIndex&, const float*, std::size_t,
                       const SearchOptions&, Neighbor*) {
    throw std::logic_error("search_std_thread is a student TODO");
}

void search_omp_student_1(const IVFIndex&, const float*, std::size_t,
                          const SearchOptions&, Neighbor*) {
    throw std::logic_error("search_omp_student_1 is a student TODO");
}

void search_omp_student_2(const IVFIndex&, const float*, std::size_t,
                          const SearchOptions&, Neighbor*) {
    throw std::logic_error("search_omp_student_2 is a student TODO");
}

}
