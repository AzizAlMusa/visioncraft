#ifndef VISIONCRAFT_VISIBILITY_MANAGER_H
#define VISIONCRAFT_VISIBILITY_MANAGER_H

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <octomap/ColorOcTree.h>
#include <boost/variant.hpp>

#include "visioncraft/model.h"


namespace visioncraft {


// Forward declaration of Viewpoint to avoid circular dependency
class Viewpoint;

/**
 * @brief Class for managing visibility tracking for multiple viewpoints.
 * 
 * The VisibilityManager class tracks visibility information of multiple viewpoints,
 * automatically managing itself as an observer of each tracked viewpoint and updating
 * visible voxel data when viewpoint raycasting results are modified.
 */
class VisibilityManager : public std::enable_shared_from_this<VisibilityManager> {
public:
    
    
    using PropertyValue = boost::variant<int, float, double, Eigen::Matrix<double, 3, 1>, std::string>;
    
    /**
     * @brief Constructor for VisibilityManager.
     * 
     * @param model A reference to the model used for visibility tracking.
     */
    explicit VisibilityManager(Model& model);

    /**
     * @brief Destructor for VisibilityManager.
     */
    ~VisibilityManager();

    /**
     * @brief Track a specific viewpoint.
     * 
     * Adds a viewpoint to the tracking list, automatically registering as an observer.
     * 
     * @param viewpoint Shared pointer to the viewpoint to track.
     */
    void trackViewpoint(const std::shared_ptr<Viewpoint>& viewpoint);

    /**
     * @brief Stop tracking a specific viewpoint.
     * 
     * Removes a viewpoint from the tracking list and unregisters as an observer.
     * 
     * @param viewpoint Shared pointer to the viewpoint to untrack.
     */
    void untrackViewpoint(const std::shared_ptr<Viewpoint>& viewpoint);

    /**
     * @brief Track multiple viewpoints.
     * 
     * Adds multiple viewpoints to the tracking list, registering as an observer for each.
     * 
     * @param viewpoints Vector of shared pointers to viewpoints to track.
     */
    void trackViewpoints(const std::vector<std::shared_ptr<Viewpoint>>& viewpoints);

    /**
     * @brief Stop tracking all viewpoints.
     * 
     * Clears the tracking list and unregisters as an observer for all viewpoints.
     */
    void untrackAllViewpoints();

    /**
     * @brief Update visibility information for a viewpoint.
     * 
     * Automatically called when any tracked viewpoint performs raycasting,
     * updating visibility data for voxels observed by that viewpoint.
     * 
     * @param viewpoint The viewpoint with updated raycasting results.
     */
    void updateVisibility(const std::shared_ptr<Viewpoint>& viewpoint);


        /**
     * @brief Get all currently visible voxels.
     * 
     * @return An unordered set of voxel keys representing visible voxels.
     */
    const std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>& getVisibleVoxels() const {
        return visible_voxels_;
    }

    /**
     * @brief Get the current visibility count for each voxel.
     * 
     * @return A const reference to the visibility count map.
     */
    const std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>& getVisibilityCount() const {
        return visibility_count_;
    }

    /**
     * @brief Get the current visibility map of viewpoints to observed voxels.
     * 
     * @return A const reference to the visibility map.
     */
    const std::unordered_map<std::shared_ptr<Viewpoint>, std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>>& getVisibilityMap() const {
        return visibility_map_;
    }

    /**
     * @brief Get the current coverage score.
     * 
     * @return The current coverage score as a double.
     */
    double getCoverageScore() const {
        return coverage_score_;
    }

    /**
     * @brief Compute and update the coverage score.
     * 
     * Calculates the percentage of all_voxels_ that are currently visible.
     * 
     * @return The current coverage score as a double.
     */
    double computeCoverageScore();

    /**
     * @brief Get the number of novel voxels seen by a specific viewpoint.
     *
     * This function calculates the number of voxels that are visible only to the specified viewpoint
     * (i.e., voxels with a visibility count of 1).
     *
     * @param viewpoint The viewpoint for which to compute the novel voxels.
     * @return The number of novel voxels seen by this viewpoint.
     */
    size_t countNovelVoxels(const std::shared_ptr<Viewpoint>& viewpoint) const;
    

    /**
     * @brief Compute the novel coverage score for a specific viewpoint.
     *
     * This function calculates the novel coverage score, which is the ratio of the number
     * of novel voxels (voxels visible only to the specified viewpoint) to the total number of
     * voxels in the model.
     *
     * @param viewpoint The viewpoint for which to compute the novel coverage score.
     * @return The novel coverage score for this viewpoint as a double.
     */
    double computeNovelCoverageScore(const std::shared_ptr<Viewpoint>& viewpoint) const;

private:


    /**
     * @brief Initializes all_voxels_ by reading keys from the model's meta voxel map.
     */
    void initializeVoxelSet();

    /**
     * @brief Updates visibility data structures when a viewpoint is added.
     * 
     * Expands visibility_map_ to include the new viewpoint and updates
     * visibility_count_ for affected voxels.
     * 
     * @param viewpoint The newly added viewpoint.
     */
    void updateVisibilityMapOnViewpointAddition(const Viewpoint& viewpoint);

    /**
     * @brief Updates visibility data structures when a viewpoint is removed.
     * 
     * Contracts visibility_map_ to exclude the viewpoint and updates
     * visibility_count_ accordingly.
     * 
     * @param viewpoint The viewpoint being removed.
     */
    void updateVisibilityMapOnViewpointRemoval(const Viewpoint& viewpoint);

    Model& model_; ///< The model used for visibility tracking.

    std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> all_voxels_; ///< Key set of all voxels of interest from the model.
    std::unordered_map<std::shared_ptr<Viewpoint>, std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>> visibility_map_; ///< Map of viewpoints to the set of visible voxel keys.
    std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash> visibility_count_; ///< Map tracking the count of viewpoints observing each voxel.
    std::unordered_set<std::shared_ptr<Viewpoint>> tracked_viewpoints_; ///< Set of all viewpoints currently being tracked.
    std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> visible_voxels_; ///< Set of voxels observed by at least one viewpoint.
    double coverage_score_; ///< Cached coverage score for quick access.

};

} // namespace visioncraft

#endif // VISIONCRAFT_VISIBILITY_MANAGER_H
