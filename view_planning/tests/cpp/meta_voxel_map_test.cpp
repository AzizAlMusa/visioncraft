// cpp/meta_voxel_map_test.cpp

#include "visioncraft/meta_voxel_map.h"
#include "visioncraft/meta_voxel.h"
#include <gtest/gtest.h>
#include <octomap/ColorOcTree.h>
#include <boost/variant.hpp>

using namespace visioncraft;

class MetaVoxelMapTest : public ::testing::Test {
protected:
    MetaVoxelMap voxel_map;
    octomap::OcTreeKey key1, key2;

    void SetUp() override {
        // Initialize test keys
        key1 = octomap::OcTreeKey(1, 2, 3);
        key2 = octomap::OcTreeKey(4, 5, 6);
    }
};

TEST_F(MetaVoxelMapTest, Constructor) {
    EXPECT_EQ(voxel_map.size(), 0);
}

TEST_F(MetaVoxelMapTest, SetAndGetMetaVoxel) {
    MetaVoxel voxel1, voxel2;
    voxel1.setProperty("density", 1.0f);
    voxel2.setProperty("temperature", 100.0f);

    EXPECT_TRUE(voxel_map.setMetaVoxel(key1, voxel1));
    EXPECT_TRUE(voxel_map.setMetaVoxel(key2, voxel2));

    MetaVoxel* retrieved_voxel1 = voxel_map.getMetaVoxel(key1);
    MetaVoxel* retrieved_voxel2 = voxel_map.getMetaVoxel(key2);

    ASSERT_NE(retrieved_voxel1, nullptr);
    ASSERT_NE(retrieved_voxel2, nullptr);

    // Check and retrieve the variant value
    auto density_value = retrieved_voxel1->getProperty("density");
    ASSERT_TRUE(density_value.type() == typeid(float));
    EXPECT_EQ(boost::get<float>(density_value), 1.0f);

    auto temperature_value = retrieved_voxel2->getProperty("temperature");
    ASSERT_TRUE(temperature_value.type() == typeid(float));
    EXPECT_EQ(boost::get<float>(temperature_value), 100.0f);
}

TEST_F(MetaVoxelMapTest, GetMetaVoxel_NotFound) {
    MetaVoxel* voxel = voxel_map.getMetaVoxel(key1);
    EXPECT_EQ(voxel, nullptr);  // Expect nullptr if key is not found
}

TEST_F(MetaVoxelMapTest, SetPropertyForAllVoxels) {
    MetaVoxel voxel1, voxel2;
    voxel_map.setMetaVoxel(key1, voxel1);
    voxel_map.setMetaVoxel(key2, voxel2);

    EXPECT_TRUE(voxel_map.setPropertyForAllVoxels("color", std::string("red")));

    MetaVoxel* retrieved_voxel1 = voxel_map.getMetaVoxel(key1);
    MetaVoxel* retrieved_voxel2 = voxel_map.getMetaVoxel(key2);

    ASSERT_NE(retrieved_voxel1, nullptr);
    ASSERT_NE(retrieved_voxel2, nullptr);

    // Check and retrieve the variant value
    auto color_value1 = retrieved_voxel1->getProperty("color");
    ASSERT_TRUE(color_value1.type() == typeid(std::string));
    EXPECT_EQ(boost::get<std::string>(color_value1), "red");

    auto color_value2 = retrieved_voxel2->getProperty("color");
    ASSERT_TRUE(color_value2.type() == typeid(std::string));
    EXPECT_EQ(boost::get<std::string>(color_value2), "red");
}

TEST_F(MetaVoxelMapTest, SetAndGetMetaVoxelProperty) {
    MetaVoxel voxel;
    voxel_map.setMetaVoxel(key1, voxel);

    EXPECT_TRUE(voxel_map.setMetaVoxelProperty(key1, "temperature", 20.5f));

    auto temperature_value = voxel_map.getMetaVoxelProperty(key1, "temperature");
    ASSERT_TRUE(temperature_value.type() == typeid(float));
    EXPECT_EQ(boost::get<float>(temperature_value), 20.5f);
}

TEST_F(MetaVoxelMapTest, GetMetaVoxelProperty_NotFound) {
    MetaVoxel voxel;
    voxel_map.setMetaVoxel(key1, voxel);

    EXPECT_THROW(voxel_map.getMetaVoxelProperty(key1, "nonexistent_property"), std::runtime_error);
}

TEST_F(MetaVoxelMapTest, Clear) {
    MetaVoxel voxel;
    voxel_map.setMetaVoxel(key1, voxel);
    voxel_map.setMetaVoxel(key2, voxel);

    EXPECT_EQ(voxel_map.size(), 2);
    voxel_map.clear();
    EXPECT_EQ(voxel_map.size(), 0);
}

TEST_F(MetaVoxelMapTest, Size) {
    MetaVoxel voxel;
    EXPECT_EQ(voxel_map.size(), 0);

    voxel_map.setMetaVoxel(key1, voxel);
    EXPECT_EQ(voxel_map.size(), 1);

    voxel_map.setMetaVoxel(key2, voxel);
    EXPECT_EQ(voxel_map.size(), 2);
}
