#ifndef VISIONCRAFT_META_VOXEL_H
#define VISIONCRAFT_META_VOXEL_H

#include <Eigen/Core>
#include <octomap/octomap.h>
#include <boost/variant.hpp>
#include <unordered_map>
#include <string>

namespace visioncraft {

/**
 * @brief MetaVoxel class represents an advanced voxel with extended properties and metadata.
 * 
 * The MetaVoxel class is designed to store a voxel's position, occupancy probability, log-odds value, 
 * and a flexible property map for additional metadata. Each MetaVoxel can hold standard occupancy-related 
 * data as well as custom properties, such as entropy or information gain, making it adaptable to 
 * various applications that extend beyond typical occupancy mapping.
 */
class MetaVoxel {
public:
    /**
     * @brief Type alias for property values, supporting multiple data types.
     * 
     * The PropertyValue type allows for the storage of diverse data types within the property map, 
     * including integer, float, Eigen::Vector3d, and string types. This flexibility enables users 
     * to extend each voxel's metadata with various attributes as needed.
     */
    using PropertyValue = boost::variant<int, float, double, Eigen::Vector3d, std::string>;

    // Default constructor
    MetaVoxel() : position_(0.0, 0.0, 0.0), occupancy_(0.5) {}

    /**
     * @brief Constructs a MetaVoxel with a specified position, OctoMap key, and initial occupancy.
     * 
     * This constructor initializes a MetaVoxel instance by setting its 3D position, OctoMap key,
     * and occupancy probability (defaulting to 0.5 if not specified). The corresponding log-odds 
     * value is automatically calculated based on the occupancy value.
     * 
     * @param position 3D position of the voxel in world coordinates, as an Eigen::Vector3d.
     * @param octomap_key OctoMap key associated with the voxel's position within an OctoMap structure.
     * @param occupancy Initial occupancy probability value for the voxel (range [0,1]), defaults to 0.5.
     */
    MetaVoxel(const Eigen::Vector3d& position, const octomap::OcTreeKey& octomap_key, float occupancy = 0.5);

    /**
     * @brief Retrieves the 3D position of the voxel.
     * 
     * This function returns the position of the voxel as a constant reference to an Eigen::Vector3d,
     * representing the voxel's location in world coordinates.
     * 
     * @return Constant reference to the voxel's 3D position.
     */
    const Eigen::Vector3d& getPosition() const;

    /**
     * @brief Retrieves the OctoMap key associated with this voxel.
     * 
     * This function provides access to the OctoMap key, allowing users to relate the MetaVoxel to 
     * corresponding entries in an OctoMap structure.
     * 
     * @return Constant reference to the OctoMap key of the voxel.
     */
    const octomap::OcTreeKey& getOctomapKey() const;

    /**
     * @brief Sets the occupancy probability and updates the log-odds.
     * 
     * This function assigns an occupancy probability to the voxel and recalculates the log-odds value
     * based on this probability, ensuring consistency between the occupancy and log-odds properties.
     * 
     * @param probability Occupancy probability (range [0,1]) to assign to the voxel.
     */
    void setOccupancy(float probability);

    /**
     * @brief Retrieves the current occupancy probability of the voxel.
     * 
     * This function returns the voxel's occupancy probability, a value within the range [0,1].
     * 
     * @return Occupancy probability of the voxel.
     */
    float getOccupancy() const;

    /**
     * @brief Sets the log-odds value and updates the occupancy probability.
     * 
     * This function assigns a log-odds value to the voxel and updates the occupancy probability
     * accordingly, ensuring consistency between the occupancy and log-odds properties.
     * 
     * @param log_odds Log-odds value to assign to the voxel.
     */
    void setLogOdds(float log_odds);

    /**
     * @brief Retrieves the current log-odds value of the voxel.
     * 
     * This function returns the voxel's log-odds value, which provides an alternative representation 
     * of occupancy that is efficient for incremental updates.
     * 
     * @return Log-odds value of the voxel.
     */
    float getLogOdds() const;

    /**
     * @brief Sets or updates a custom property in the voxel's property map.
     * 
     * This function allows for the addition of custom properties to the voxel's property map.
     * Properties can be defined as needed, making the voxel adaptable to various use cases.
     * 
     * @param property_name Name of the property to add or update in the property map.
     * @param value Value to assign to the property, using the PropertyValue type.
     */
    void setProperty(const std::string& property_name, const PropertyValue& value);

    /**
     * @brief Retrieves a specified property from the voxel's property map.
     * 
     * This function fetches the value of a specified property from the voxel's property map.
     * If the property does not exist, it throws a runtime exception.
     * 
     * @param property_name Name of the property to retrieve.
     * @return Value of the specified property as a PropertyValue type.
     * @throws std::runtime_error If the property does not exist in the property map.
     */
    PropertyValue getProperty(const std::string& property_name) const;

    /**
     * @brief Checks for the existence of a specified property in the voxel's property map.
     * 
     * This function verifies whether a property with the given name exists in the property map.
     * 
     * @param property_name Name of the property to check for in the map.
     * @return True if the property exists, false otherwise.
     */
    bool hasProperty(const std::string& property_name) const;

private:
    Eigen::Vector3d position_;  ///< 3D position of the voxel in world coordinates.
    octomap::OcTreeKey octomap_key_;  ///< OctoMap key linking the voxel to an OctoMap structure.
    std::unordered_map<std::string, PropertyValue> properties_;  ///< Customizable property map for dynamic metadata.
    float occupancy_;  ///< Occupancy probability, a value between 0 (free) and 1 (occupied).
    float log_odds_;   ///< Log-odds representation of occupancy, providing efficient incremental updates.

    /**
     * @brief Calculates the log-odds value from an occupancy probability.
     * 
     * This helper function computes the log-odds corresponding to a given occupancy probability,
     * using the formula: log_odds = log(occupancy / (1 - occupancy)).
     * 
     * @param probability Occupancy probability to convert.
     * @return Log-odds value calculated from the occupancy probability.
     */
    static float occupancyToLogOdds(float probability);

    /**
     * @brief Calculates the occupancy probability from a log-odds value.
     * 
     * This helper function computes the occupancy probability corresponding to a given log-odds value,
     * using the formula: occupancy = 1 / (1 + exp(-log_odds)).
     * 
     * @param log_odds Log-odds value to convert.
     * @return Occupancy probability calculated from the log-odds value.
     */
    static float logOddsToOccupancy(float log_odds);

}; // class MetaVoxel

} // namespace visioncraft

#endif // VISIONCRAFT_META_VOXEL_H
