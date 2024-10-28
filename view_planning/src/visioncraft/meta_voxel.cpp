#include "meta_voxel.h"
#include <stdexcept>
#include <cmath>

namespace visioncraft {

/**
 * @brief Constructs a MetaVoxel with specified position, OctoMap key, and initial occupancy.
 * 
 * The constructor initializes the MetaVoxel, storing the 3D position, OctoMap key, and occupancy.
 * The log-odds value is computed based on the provided occupancy.
 * 
 * @param position 3D position of the voxel as an Eigen::Vector3d.
 * @param octomap_key Key corresponding to the voxel’s position in the OctoMap.
 * @param occupancy Initial occupancy probability (default is 0.5).
 */
MetaVoxel::MetaVoxel(const Eigen::Vector3d& position, const octomap::OcTreeKey& octomap_key, float occupancy)
    : position_(position), octomap_key_(octomap_key), occupancy_(occupancy), log_odds_(occupancyToLogOdds(occupancy)) {}

/**
 * @brief Retrieve the voxel's 3D position.
 * 
 * @return The voxel's position as an Eigen::Vector3d.
 */
const Eigen::Vector3d& MetaVoxel::getPosition() const {
    return position_;
}

/**
 * @brief Retrieve the OctoMap key associated with this voxel.
 * 
 * @return The voxel's OctoMap key as an octomap::OcTreeKey.
 */
const octomap::OcTreeKey& MetaVoxel::getOctomapKey() const {
    return octomap_key_;
}

/**
 * @brief Set the occupancy probability and update the log-odds.
 * 
 * This method sets the occupancy probability and computes the log-odds accordingly.
 * 
 * @param probability Occupancy probability value.
 */
void MetaVoxel::setOccupancy(float probability) {
    occupancy_ = probability;
    log_odds_ = occupancyToLogOdds(probability);
}

/**
 * @brief Get the current occupancy probability.
 * 
 * @return Occupancy probability as a float.
 */
float MetaVoxel::getOccupancy() const {
    return occupancy_;
}

/**
 * @brief Set the log-odds and update the occupancy probability.
 * 
 * This method sets the log-odds value and updates the occupancy probability accordingly.
 * 
 * @param log_odds Log-odds value.
 */
void MetaVoxel::setLogOdds(float log_odds) {
    log_odds_ = log_odds;
    occupancy_ = logOddsToOccupancy(log_odds);
}

/**
 * @brief Get the current log-odds value.
 * 
 * @return Log-odds value as a float.
 */
float MetaVoxel::getLogOdds() const {
    return log_odds_;
}

// Static helper function: Convert occupancy probability to log-odds
float MetaVoxel::occupancyToLogOdds(float probability) {
    return std::log(probability / (1.0f - probability));
}

// Static helper function: Convert log-odds to occupancy probability
float MetaVoxel::logOddsToOccupancy(float log_odds) {
    return 1.0f / (1.0f + std::exp(-log_odds));
}

/**
 * @brief Set or update a property in the voxel’s property map.
 * 
 * Adds a new property or updates an existing one, using the specified property name and value.
 * 
 * @param property_name Name of the property to add or update.
 * @param value Value of the property, as a PropertyValue type.
 */
void MetaVoxel::setProperty(const std::string& property_name, const PropertyValue& value) {
    properties_[property_name] = value;
}

/**
 * @brief Retrieve the value of a specified property.
 * 
 * Fetches the property value from the map. Throws an exception if the property does not exist.
 * 
 * @param property_name Name of the property to retrieve.
 * @return Value of the specified property as a PropertyValue type.
 * @throws std::runtime_error If the property does not exist in the map.
 */
MetaVoxel::PropertyValue MetaVoxel::getProperty(const std::string& property_name) const {
    auto it = properties_.find(property_name);
    if (it != properties_.end()) {
        return it->second;
    } else {
        throw std::runtime_error("Property '" + property_name + "' not found in MetaVoxel.");
    }
}

/**
 * @brief Check for the existence of a specified property.
 * 
 * This method verifies if a property with the given name exists in the property map.
 * 
 * @param property_name Name of the property to check.
 * @return True if the property exists, false otherwise.
 */
bool MetaVoxel::hasProperty(const std::string& property_name) const {
    return properties_.find(property_name) != properties_.end();
}

} // namespace visioncraft
