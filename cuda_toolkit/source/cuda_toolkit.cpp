//
//#include "cuda_session.h"
//#include "cuda_toolkit.hpp"
//
//
//static CudaManager& get_session()
//{
//    static CudaManager session;
//    return session;
//}
//
//
//bool 
//cuda_toolkit::beamform(std::span<const uint8_t> input_data, 
//                          std::span<uint8_t> output_data, 
//                          const CudaBeamformerParameters& bp)
//{
//    auto& session = get_session();
//
//    if (!session.init({bp.rf_raw_dim[0], bp.rf_raw_dim[1]}, 
//                      {bp.dec_data_dim[0], bp.dec_data_dim[1], bp.dec_data_dim[2]}))
//    {
//        std::cerr << "Failed to initialize CUDA session." << std::endl;
//        return false;
//    }
//
//
//    return true;
//}
