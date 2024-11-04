#include "visioncraft/meta_voxel_map.h"
#include <iostream>


#include <execinfo.h>

namespace visioncraft {

/**
 * @brief Constructor for MetaVoxelMap.
 */
MetaVoxelMap::MetaVoxelMap() {
    // std::cout << "MetaVoxelMap constructor called." << std::endl;
}

/**
 * @brief Destructor for MetaVoxelMap.
 */
MetaVoxelMap::~MetaVoxelMap() {
    // std::cout << "MetaVoxelMap destructor called." << std::endl;
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
MetaVoxel* MetaVoxelMap::getMetaVoxel(const octomap::OcTreeKey& key) const {
    // Attempt to find the key in meta_voxel_map_
    auto it = meta_voxel_map_.find(key);

    if (it != meta_voxel_map_.end()) {
        return const_cast<MetaVoxel*>(&it->second);
    } else {
        // Print the key being searched for
        std::cerr << "MetaVoxel not found for key: (" << key.k[0] << ", " << key.k[1] << ", " << key.k[2] << ")" << std::endl;

        // Debug: Iterate manually to check for close matches
        for (const auto& entry : meta_voxel_map_) {
            const octomap::OcTreeKey& stored_key = entry.first;
            if (stored_key.k[0] == key.k[0] && stored_key.k[1] == key.k[1] && stored_key.k[2] == key.k[2]) {
                std::cerr << "Potential match found with identical values in stored key, but find() failed. Stored Key: ("
                          << stored_key.k[0] << ", " << stored_key.k[1] << ", " << stored_key.k[2] << ")" << std::endl;
                break;
            }
        }

        return nullptr;
    }
}



/**
 * @brief Set a specified property for all MetaVoxel instances within the map.
 * 
 * This function iterates over each MetaVoxel in the internal map and assigns the specified 
 * property with the provided value. It ensures that each voxel has the same property, which 
 * can be useful for initializing or updating attributes uniformly across all voxels.
 * 
 * @param property_name The name of the property to set for each MetaVoxel (e.g., "temperature").
 * @param value The value to assign to the property for all MetaVoxels, utilizing the PropertyValue type.
 * @return True if the property is successfully set for all MetaVoxels, false if the map is empty or operation fails.
 */
bool MetaVoxelMap::setPropertyForAllVoxels(const std::string& property_name, const MetaVoxel::PropertyValue& value) {
    for (auto& kv : meta_voxel_map_) {
        kv.second.setProperty(property_name, value);
    }
    return true; // return true if the operation completes successfully
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
