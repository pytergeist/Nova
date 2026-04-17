//
//TEST(AutodiffMetaTest, construct_meta_binary_produces_two_tensor_meta) {
//      RawTensor<float> t1({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
//                      DType::FLOAT32, Device{DeviceType::CPU, 0});
//      RawTensor<float> t2({3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6},
//                      DType::FLOAT32, Device{DeviceType::CPU, 0});
//
//      AutodiffMeta<float> meta = construct_meta<float>(t1, t2);
//      EXPECT_EQ(meta.size(), 2);
//      EXPECT_NE(&meta[0], &t1);
//      EXPECT_NE(&meta[1], &t2);
//
//      EXPECT_EQ(meta[0].shape(), t1.shape());
//      EXPECT_EQ(meta[1].shape(), t2.shape());
//}