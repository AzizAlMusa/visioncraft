#include "visioncraft/meta_voxel_map.h"
#include <iostream>

namespace visioncraft {

/**
 * @brief Constructor for MetaVoxelMap.
 */
MetaVoxelMap::MetaVoxelMap() {
    std::cout << "MetaVoxelMap constructor called." << std::endl;
}

/**
 * @brief Destructor for MetaVoxelMap.
 */
MetaVoxelMap::~MetaVoxelMap() {
    std::cout << "MetaVoxelMap destructor called." << std::endl;
}

/**
 * @brief Insert or update a MetaVoxel object in the map.
 *
 * This function either inserts a new MetaVoxel or updates an existing one 
 * if a MetaVoxel with the specified key already exists.
 * 
 * @param key The OctoMap key for spatial indexing of the MetaVoxel.
 * @param meta_voxel The MetaVoxel instance to insert or update.
 * @return True if the operation is successful, false otherwise.
 */
bool MetaVoxelMap::setMetaVoxel(const octomap::OcTreeKey& key, const MetaVoxel& meta_voxel) {
    meta_voxel_map_[key] = meta_voxel;
    return true;
}

/**
 * @brief Retrieve a MetaVoxel instance from the map.
 * 
 * This function retrieves the MetaVoxel instance associated with the given OctoMap key.
 * 
 * @param key The OctoMap key for spatial indexing.
 * @return Pointer to the MetaVoxel if found, nullptr otherwise.
 */
MetaVoxel* MetaVoxelMap::getMetaVoxel(const octomap::OcTreeKey& key) {
    auto it = meta_voxel_map_.find(key);
    if (it != meta_voxel_map_.end()) {
        return &(it->second);
    } else {
        std::cerr << "MetaVoxel not found for the provided key." << std::endl;
        return nullptr;
    }
}

/**
 * @brief Set a custom property for a specific MetaVoxel.
 * 
 * This function sets a custom property for a MetaVoxel identified by its key. If the MetaVoxel
 * does not exist, it returns false.
 * 
 * @param key The OctoMap key for spatial indexing.
 * @param property_name The name of the property to set.
 * @param value The value to assign to the property.
 * @return True if the property is set successfully, false otherwise.
 */
bool MetaVoxelMap::setMetaVoxelProperty(const octomap::OcTreeKey& key, const std::string& property_name, const MetaVoxel::PropertyValue& value) {
    MetaVoxel* voxel = getMetaVoxel(key);
    if (voxel) {
        voxel->setProperty(property_name, value);
        return true;
    } else {
        return false;
    }
}

/**
 * @brief Retrieve a custom property from a specific MetaVoxel.
 * 
 * This function retrieves a specified custom property from a MetaVoxel, identified by its key.
 * 
 * @param key The OctoMap key for spatial indexing.
 * @param property_name The name of the property to retrieve.
 * @return The property value if found, throws runtime_error if not found.
 */
MetaVoxel::PropertyValue MetaVoxelMap::getMetaVoxelProperty(const octomap::OcTreeKey& key, const std::string& property_name) const {
    auto it = meta_voxel_map_.find(key);
    if (it != meta_voxel_map_.end()) {
        return it->second.getProperty(property_name);
    } else {
        throw std::runtime_error("MetaVoxel or property not found for the provided key.");
    }
}

/**
 * @brief Clear all MetaVoxel entries in the map.
 * 
 * This function removes all MetaVoxel instances from the map.
 */
void MetaVoxelMap::clear() {
    meta_voxel_map_.clear();
    std::cout << "MetaVoxelMap cleared." << std::endl;
}

/**
 * @brief Get the size of the MetaVoxel map.
 * 
 * @return The number of MetaVoxel entries in the map.
 */
size_t MetaVoxelMap::size() const {
    return meta_voxel_map_.size();
}

} // namespace visioncraft
