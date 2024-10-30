#include <gtest/gtest.h>
#include "meta_voxel.h"

using namespace visioncraft;

// Fixture class for setting up common data for MetaVoxel tests
class MetaVoxelTest : public ::testing::Test {
protected:
    // Common test setup variables
    Eigen::Vector3d sample_position;
    octomap::OcTreeKey sample_key;
    float sample_occupancy;
    MetaVoxel meta_voxel;

    MetaVoxelTest() 
        : sample_position(1.0, 2.0, 3.0),
          sample_key{100, 200, 300},
          sample_occupancy(0.5),
          meta_voxel(sample_position, sample_key, sample_occupancy) {}

    // Test helper to check property existence and value
    template <typename T>
    void checkProperty(const MetaVoxel& voxel, const std::string& name, const T& expected_value) {
        ASSERT_TRUE(voxel.hasProperty(name));
        auto actual_value = boost::get<T>(voxel.getProperty(name));
        ASSERT_EQ(actual_value, expected_value);
    }
};

TEST_F(MetaVoxelTest, Constructor) {
    // Ensure that the constructor initializes correctly
    EXPECT_EQ(meta_voxel.getPosition(), sample_position);
    EXPECT_EQ(meta_voxel.getOctomapKey(), sample_key);
    EXPECT_FLOAT_EQ(meta_voxel.getOccupancy(), sample_occupancy);
}

TEST_F(MetaVoxelTest, SetAndGetPosition) {
    // Test retrieving position
    ASSERT_EQ(meta_voxel.getPosition(), sample_position);
}

TEST_F(MetaVoxelTest, SetAndGetOccupancy) {
    // Set and test new occupancy
    float new_occupancy = 0.8;
    meta_voxel.setOccupancy(new_occupancy);
    EXPECT_FLOAT_EQ(meta_voxel.getOccupancy(), new_occupancy);
}

TEST_F(MetaVoxelTest, SetAndGetLogOdds) {
    // Test setting and retrieving log-odds, check occupancy consistency
    float log_odds_value = -0.2;
    meta_voxel.setLogOdds(log_odds_value);
    EXPECT_FLOAT_EQ(meta_voxel.getLogOdds(), log_odds_value);

    float expected_occupancy = 1.0 / (1.0 + std::exp(-log_odds_value));
    EXPECT_NEAR(meta_voxel.getOccupancy(), expected_occupancy, 1e-6);
}

TEST_F(MetaVoxelTest, SetAndGetCustomProperties) {
    // Integer property
    meta_voxel.setProperty("temperature", 25);
    checkProperty(meta_voxel, "temperature", 25);

    // Float property
    meta_voxel.setProperty("pressure", 101.3f);
    checkProperty(meta_voxel, "pressure", 101.3f);

    // String property
    std::string status = "active";
    meta_voxel.setProperty("status", status);
    checkProperty(meta_voxel, "status", status);

    // Vector property
    Eigen::Vector3d velocity(0.5, 1.5, -0.5);
    meta_voxel.setProperty("velocity", velocity);
    ASSERT_TRUE(meta_voxel.hasProperty("velocity"));
    ASSERT_EQ(boost::get<Eigen::Vector3d>(meta_voxel.getProperty("velocity")), velocity);
}

TEST_F(MetaVoxelTest, HasProperty) {
    // Test checking property existence
    meta_voxel.setProperty("temperature", 25);
    ASSERT_TRUE(meta_voxel.hasProperty("temperature"));
    ASSERT_FALSE(meta_voxel.hasProperty("nonexistent_property"));
}

TEST_F(MetaVoxelTest, GetNonExistentPropertyThrows) {
    // Test that retrieving a non-existent property throws an error
    ASSERT_THROW(meta_voxel.getProperty("nonexistent_property"), std::runtime_error);
}

// Main runner for all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
