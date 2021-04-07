#pragma once

#include <type_traits>
#include <iterator>
#include <memory> //unique_ptr
#include <vector>
#include <cassert>

//debugging
#include <array>
#include <list>
#include <iostream>



#include <tensorflow/c/c_api.h>

namespace tfplusplus {

using TF_Tensor_ptr =
  std::unique_ptr< TF_Tensor, TF_DeleteTensor >;

namespace tfplusplus_devel {

namespace type_deduction {

template< typename T >
struct TFDataTypeDetector {
  //This is only called if there is no special instantiation below
  TF_DataType value = TF_FLOAT;

  static void validate(){
    assert( false );
  }
};

template<>
struct TFDataTypeDetector< float > {
  TF_DataType value = TF_FLOAT;

  static void validate(){}
};

template<>
struct TFDataTypeDetector< double > {
  TF_DataType value = TF_DOUBLE;

  static void validate(){}
};

template<>
struct TFDataTypeDetector< bool > {
  TF_DataType value = TF_BOOL;

  static void validate(){}
};

template<>
struct TFDataTypeDetector< int > {
  TF_DataType value = TF_INT32;

  static void validate(){ runtime_assert( sizeof(int) == 4 ); }
};

template<>
struct TFDataTypeDetector< unsigned int > {
  TF_DataType value = TF_UINT32;

  static void validate(){ runtime_assert( sizeof(unsigned int) == 4 ); }
};

template<>
struct TFDataTypeDetector< long int > {
  TF_DataType value = TF_INT64;

  static void validate(){ runtime_assert( sizeof(long int) == 8 ); }
};

template<>
struct TFDataTypeDetector< unsigned long int > {
  TF_DataType value = TF_UINT64;

  static void validate(){ runtime_assert( sizeof(unsigned long int) == 8 ); }
};

}

template <typename T>
concept arithmetic = std::is_arithmetic< T >::value;

template< typename T >
int64_t
get_num_values( T const & t ){
  if constexpr ( arithmetic< T > ) {
    return 1;
  }
  else {
    static_assert( std::ranges::range<T>, "Must pass a range of arithmetic" );
    assert( ! t.empty() );
    return t.size() * get_num_values< typename T::value_type >( * t.begin() );
  }
}

template< typename T >
constexpr
int
get_num_dims(){
  if constexpr ( arithmetic< T > ) {
    return 0;
  }
  else {
    static_assert( std::ranges::range<T>, "Must pass a range of arithmetic" );
    return 1 + get_num_dims< typename T::value_type >();
  }
}

template< int NDIM, typename T >
void
get_dims(
  T const & t,
  std::array< int, NDIM > & dims,
  int const current_dim = 0
){
  if constexpr ( arithmetic< T > ) {
    return;
  }
  else {
    static_assert( std::ranges::range<T>, "Must pass a range of arithmetic" );

    using category = typename std::iterator_traits< typename T::iterator >::iterator_category;
    static_assert( std::is_same_v<category, std::random_access_iterator_tag>,
      "We only support random access iterators right now" );

    assert( ! t.empty() );
    dims[ current_dim ] = t.size();

    get_dims< NDIM, typename T::value_type >( *t.begin(), dims, current_dim + 1 );
  }
}

template< int NDIM, typename T >
std::array< int, NDIM >
get_dims( T const & t ) {
  std::array< int, NDIM > dims;
  get_dims< NDIM >( t, dims, 0 );
  return dims;
}

/*template< typename T >
float const *
get_first_float_ptr( T const & t ){
  static_assert( std::ranges::range<T>, "Must pass a range of arithmetic" );
  if constexpr( std::ranges::range< typename T::value_type > ) {
    assert( ! t.empty() );
    return get_first_float_ptr< typename T::value_type >( * t.begin() );
  } else {
    return &(*t.begin());
  }
}*/

template< typename T >
auto const *
get_first_value_ptr( T const & t ){
  static_assert( std::ranges::range<T>, "Must pass a range of arithmetic" );
  if constexpr( std::ranges::range< typename T::value_type > ) {
    assert( ! t.empty() );
    return get_first_value_ptr< typename T::value_type >( * t.begin() );
  } else {
    return &(*t.begin());
  }
}


/*template< typename T >
constexpr
bool
has_float_value_type(){
  if constexpr( arithmetic< T > ){
    return std::is_same_v< T, float >;
  } else {
    static_assert( std::ranges::range< T > );
    return has_float_value_type< typename T::value_type >();
  }
}*/

template< typename T, typename EXPECTED >
constexpr
bool
has_expected_value_type(){
  if constexpr( arithmetic< T > ){
    return std::is_same_v< T, EXPECTED >;
  } else {
    static_assert( std::ranges::range< T > );
    return has_expected_value_type< typename T::value_type, EXPECTED >();
  }
}

template< typename T, typename ValType >
void
get_values(
  T const & t,
  std::vector< ValType > & values
){
  if constexpr ( arithmetic< typename T::value_type > ) {
    static_assert( false, "Dead Code?" );
    //We are at the bottom layer
    return;
  }
  else {
    static_assert( std::ranges::range<T>, "Must pass a range of arithmetic" );
    static_assert( has_expected_value_type< T, ValType >() );

    assert( ! t.empty() );

    if constexpr( std::is_trivially_copyable< typename T::value_type >::value ){
      //Can do copying here
      //std::cout << "CAN COPY" << std::endl;
      ValType const * data = get_first_value_ptr( t );
      int64_t const nval_to_copy = get_num_values( t );
      values.insert( values.end(), data, data + nval_to_copy );
      return;
    } else {
      //std::cout << "NO COPY" << std::endl;
      for( auto & subrange : t ){
	get_values< typename T::value_type >( subrange, values );
      }
    }
  }

}

} //namespace tfplusplus_devel

//From stack overflow
/*TF_Tensor *
FloatTensor(
  int64_t const * dims,
  int const num_dims,
  int64_t const num_values,
  float const * values
) {
  TF_Tensor *t =
    TF_AllocateTensor(TF_FLOAT, dims, num_dims, sizeof(float) * num_values);

  memcpy(TF_TensorData(t), values, sizeof(float) * num_values);

  return t;
}*/

template< typename Container, typename ValType >
TF_Tensor_ptr
run( Container const & data4tensor ){
  static_assert( std::ranges::range< Container >, "Must pass a range of arithmetic" );

  using namespace tfplusplus_devel;
  using namespace tfplusplus_devel::type_deduction;

  constexpr int NDIM = get_num_dims< Container >();
  std::array< int, NDIM > const dims = get_dims( data4tensor );

  int64_t const nvalues = get_num_values( data4tensor );
  std::vector< ValType > values;
  values.reserve( nvalues );
  get_values( data4tensor, values );
  assert( values.size() == nvalues );

  //Make Tensor
  TFDataTypeDetector< ValType >::validate();
  TF_Tensor * tensor = TF_AllocateTensor(
    TFDataTypeDetector< ValType >::value,
    dims.data(), NDIM, sizeof( ValType ) * nvalues
  );

  assert( TF_TensorByteSize( tensor ) == sizeof( ValType ) * nvalues );
  memcpy( TF_TensorData( tensor ), values.data(), sizeof( ValType ) * nvalues );

  return TF_Tensor_ptr( tensor );
}

} //namespace tfplusplus
