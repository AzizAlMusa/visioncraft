#include "visioncraft/viewpoint.h"
#include "visioncraft/meta_voxel_map.h"
#include "visioncraft/visibility_manager.h"

#include <unordered_map>
#include <set>
#include <iterator>

namespace visioncraft {

VisibilityManager::VisibilityManager(Model& model) 
    : model_(model), all_voxels_(), visibility_map_() {
    
    // Initialize all_voxels_ by extracting keys from the model's meta_voxel_map_
    const auto& meta_voxel_map = model_.getVoxelMap().getMap();
    for (const auto& pair : meta_voxel_map) {
        all_voxels_.insert(pair.first);  // Insert each voxel key from meta_voxel_map_ into all_voxels_
    }

    // add a visibility property for the voxels
    model_.addVoxelProperty("visibility", 0);
    // visibility_map_ is left empty here, to be populated later as viewpoints are tracked
}


VisibilityManager::~VisibilityManager() {
    untrackAllViewpoints();
}

void VisibilityManager::trackViewpoint(const std::shared_ptr<Viewpoint>& viewpoint) {
    if (viewpoint && tracked_viewpoints_.insert(viewpoint).second) {
        viewpoint->addObserver(shared_from_this());

        // Initialize an empty set for the new viewpoint in visibility_map_
        visibility_map_.emplace(viewpoint, std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>());
    }
}

void VisibilityManager::untrackViewpoint(const std::shared_ptr<Viewpoint>& viewpoint) {

    if (tracked_viewpoints_.erase(viewpoint) > 0) {
        viewpoint->removeObserver(shared_from_this());

        // Remove the viewpoint entry from visibility_map_ only if the shared_ptr key matches
        auto it = visibility_map_.find(viewpoint);
        if (it != visibility_map_.end()) {
            for (const auto& voxel : it->second) {
                visible_voxels_.erase(voxel);  // Remove from visible_voxels_ if no longer tracked
            }
            visibility_map_.erase(it);  // Remove viewpoint from visibility_map_
        }
    }
}


void VisibilityManager::trackViewpoints(const std::vector<std::shared_ptr<Viewpoint>>& viewpoints) {
    for (const auto& viewpoint : viewpoints) {
        trackViewpoint(viewpoint);
    }
}

void VisibilityManager::untrackAllViewpoints() {
    for (const auto& viewpoint : tracked_viewpoints_) {
        viewpoint->removeObserver(shared_from_this());
    }
    tracked_viewpoints_.clear();
    visibility_map_.clear();  
}


void VisibilityManager::updateVisibility(const std::shared_ptr<Viewpoint>& viewpoint) {
    // Retrieve the set of voxels currently visible to this viewpoint
    auto& viewpoint_visibility_set = visibility_map_[viewpoint];

    // Decrement the visibility count for each voxel currently in viewpoint_visibility_set
    for (const auto& voxel : viewpoint_visibility_set) {
        if (--visibility_count_[voxel] == 0) {
            // If the visibility count drops to zero, remove it from visible_voxels_
            visible_voxels_.erase(voxel);
            visibility_count_.erase(voxel);  // Clean up visibility count for the voxel
        }
    }

    // Clear the old hits for this viewpoint in visibility_map_
    viewpoint_visibility_set.clear();

    // Process new hits
    const auto& hits = viewpoint->getHitResults();
    for (const auto& hit : hits) {
        if (hit.second) {
            // Update visibility set and counters for new hits
            viewpoint_visibility_set.insert(hit.first);
            visible_voxels_.insert(hit.first);
            visibility_count_[hit.first]++;
        }
    }

    computeCoverageScore();
}

double VisibilityManager::computeCoverageScore() {
    // Compute the ratio of visible voxels to all voxels
    if (all_voxels_.empty()) {
        coverage_score_ = 0.0;
    } else {
        coverage_score_ = static_cast<double>(visible_voxels_.size()) / all_voxels_.size();
    }
    return coverage_score_;
}



} // namespace visioncraft
