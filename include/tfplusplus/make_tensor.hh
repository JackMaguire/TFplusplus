#include <type_traits>
#include <iterator>
#include <iostream>

#include <vector>
#include <array>
#include <list>

#include <cassert>

#include <tensorflow/c/c_api.h>

namespace tfplusplus {

template <typename T>
concept arithmetic = std::is_arithmetic< T >::value;

template< typename T >
void print_shape( T const & t ){
  if constexpr ( arithmetic< T > ) {
    std::cout << " )";
    return;
  }
  else {
    static_assert( std::ranges::range<T>, "Must pass a range of arithmetic" );

    using category = typename std::iterator_traits< typename T::iterator >::iterator_category;
    static_assert( std::is_same_v<category, std::random_access_iterator_tag>,
      "We only support random access iterators right now" );

    std::cout << ' ' << t.size();

    if constexpr ( arithmetic< typename T::value_type > ) {
      std::cout << " )";
      return;
    }
    else {
      std::cout << ',';
      print_shape< typename T::value_type >( t[0] );
    }
  }
    
}

template< typename T >
int64_t
get_num_values( T const & t ){
  if constexpr ( arithmetic< T > ) {
    return 1;
  }
  else {
    static_assert( std::ranges::range<T>, "Must pass a range of arithmetic" );

    //using category = typename std::iterator_traits< typename T::iterator >::iterator_category;
    //static_assert( std::is_same_v<category, std::random_access_iterator_tag>,
    //  "We only support random access iterators right now" );

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
  
    get_dims< NDIM, typename T::value_type >( t[0], dims, current_dim + 1 );
  }
}

template< int NDIM, typename T >
std::array< int, NDIM >
get_dims( T const & t ) {
  std::array< int, NDIM > dims;
  get_dims< NDIM >( t, dims, 0 );
  return dims;
}

template< typename T >
float const *
get_first_float_ptr( T const & t ){
  static_assert( std::ranges::range<T>, "Must pass a range of arithmetic" );
  if constexpr( std::ranges::range< typename T::value_type > ) {
    assert( ! t.empty() );
    return get_first_float_ptr< typename T::value_type >( * t.begin() );
  } else {
    return &(*t.begin());
  }
}

template< typename T >
constexpr
bool
has_float_value_type(){
  if constexpr( arithmetic< T > ){
    return std::is_same_v< T, float >;
  } else {
    static_assert( std::ranges::range< T > );
    return has_float_value_type< typename T::value_type >();
  }
}

template< typename T >
void
get_values(
  T const & t,
  std::vector< float > & values
){
  static_assert( has_float_value_type< T >(), "assumes float" );

  if constexpr ( arithmetic< typename T::value_type > ) {
    static_assert( false, "Dead Code?" );
    //We are at the bottom layer
    return;
  }
  else {
    static_assert( std::ranges::range<T>, "Must pass a range of arithmetic" );

    using category = typename std::iterator_traits< typename T::iterator >::iterator_category;
    static_assert( std::is_same_v<category, std::random_access_iterator_tag>,
      "We only support random access iterators right now" );

    assert( ! t.empty() );

    if constexpr( std::is_trivially_copyable< typename T::value_type >::value ){
      //Can do copying here
      //std::cout << "CAN COPY" << std::endl;
      float const * data = get_first_float_ptr( t );
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

template< typename Container >
TF_Tensor *
run( Container const & data4tensor ){
  static_assert( std::ranges::range< Container >, "Must pass a range of arithmetic" );

  constexpr int NDIM = get_num_dims< Container >();
  std::array< int, NDIM > const dims = get_dims( data4tensor );

  int64_t const nvalues = get_num_values( data4tensor );
  std::vector< float > values;
  get_values( data4tensor, values );
  assert( values.size() == nvalues );

  //Make Tensor
  TF_Tensor * tensor = TF_AllocateTensor( TF_FLOAT, dims.data(), NDIM, sizeof(float) * nvalues );
  memcpy( TF_TensorData( tensor ), values.data(), sizeof(float) * nvalues );
  
}

} //namespace tfplusplus

#define runmain
#ifdef runmain

int main(){
  /*int i = 0;
  run( i );

  float f = 0;
  run( f );

  std::vector< int > vi;
  run( vi );

  std::list< int > li;
  run( li );

  std::array< int, 5 > ai;
  run( ai );
  std::cout << '('; print_shape( ai ); std::cout << std::endl;

  std::array< std::array< int, 15 >, 5 > aai;
  run( aai );
  std::cout << '('; print_shape( aai ); std::cout << std::endl;

  std::vector< std::array< int, 15 > > vai( 10 );
  run( vai );
  std::cout << '('; print_shape( vai ); std::cout << std::endl;

  std::list< std::array< int, 15 > > lai( 10 );
  run( lai );
  //std::cout << '('; print_shape( lai ); std::cout << std::endl;*/
}

#endif
