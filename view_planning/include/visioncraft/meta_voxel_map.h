#ifndef VISIONCRAFT_META_VOXEL_MAP_H
#define VISIONCRAFT_META_VOXEL_MAP_H

#include <unordered_map>
#include <string>
#include <stdexcept>
#include <octomap/ColorOcTree.h>
#include "visioncraft/meta_voxel.h"

namespace visioncraft {

/**
 * @brief Class for managing MetaVoxel instances within a map structure.
 *
 * This class provides efficient access and management of MetaVoxel instances, storing
 * them with OctoMap keys for spatial indexing. It offers methods to retrieve, update, 
 * and manage properties within each MetaVoxel instance.
 */
class MetaVoxelMap {
public:
    /**
     * @brief Constructor for MetaVoxelMap.
     */
    MetaVoxelMap();

    /**
     * @brief Destructor for MetaVoxelMap.
     */
    ~MetaVoxelMap();

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
    bool setMetaVoxel(const octomap::OcTreeKey& key, const MetaVoxel& meta_voxel);

    /**
     * @brief Retrieve a MetaVoxel instance from the map.
     * 
     * This function retrieves the MetaVoxel instance associated with the given OctoMap key.
     * 
     * @param key The OctoMap key for spatial indexing.
     * @return Pointer to the MetaVoxel if found, nullptr otherwise.
     */
    MetaVoxel* getMetaVoxel(const octomap::OcTreeKey& key);

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
    bool setMetaVoxelProperty(const octomap::OcTreeKey& key, const std::string& property_name, const MetaVoxel::PropertyValue& value);

    /**
     * @brief Retrieve a custom property from a specific MetaVoxel.
     * 
     * This function retrieves a specified custom property from a MetaVoxel, identified by its key.
     * 
     * @param key The OctoMap key for spatial indexing.
     * @param property_name The name of the property to retrieve.
     * @return The property value if found, throws runtime_error if not found.
     */
    MetaVoxel::PropertyValue getMetaVoxelProperty(const octomap::OcTreeKey& key, const std::string& property_name) const;

    /**
     * @brief Clear all MetaVoxel entries in the map.
     * 
     * This function removes all MetaVoxel instances from the map.
     */
    void clear();

    /**
     * @brief Get the size of the MetaVoxel map.
     * 
     * @return The number of MetaVoxel entries in the map.
     */
    size_t size() const;

    /**
     * @brief Retrieve the internal meta voxel map.
     *
     * @return A const reference to the unordered map of MetaVoxel objects.
     */
    const std::unordered_map<octomap::OcTreeKey, MetaVoxel, octomap::OcTreeKey::KeyHash>& getMap() const {
        return meta_voxel_map_;
    }

private:
    std::unordered_map<octomap::OcTreeKey, MetaVoxel, octomap::OcTreeKey::KeyHash> meta_voxel_map_; ///< Map of MetaVoxel objects, keyed by OctoMap keys.
};

} // namespace visioncraft

#endif // VISIONCRAFT_META_VOXEL_MAP_H
