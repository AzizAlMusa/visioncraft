std::vector<std::pair<bool, Eigen::Vector3d>> Viewpoint::performRaycasting(
    const std::shared_ptr<octomap::ColorOcTree>& octomap, bool use_parallel) {
    
    // If rays have not been generated yet, generate them.
    if (rays_.empty()) {
        rays_ = generateRays(); // Generate rays based on the current viewpoint parameters.
    }

    // Prepare a vector to store the results of the raycasting.
    // Each element in the vector will be a pair, where the first element is a boolean (hit or miss),
    // and the second element is the 3D coordinates of the hit point (or a zero vector if no hit).
    std::vector<std::pair<bool, Eigen::Vector3d>> hit_results(rays_.size());

    // Define light green color (you can adjust the intensity here)
    const uint8_t light_green_r = 144;
    const uint8_t light_green_g = 238;
    const uint8_t light_green_b = 144;

    // Define the increment to add to the green component per hit
    const uint8_t green_increment = 30;

    // Check if multithreading is enabled
    if (use_parallel) {
        // Multithreading version: split the work into multiple threads

        // Determine the number of available hardware threads.
        const int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;  // Vector to store the threads.
        std::vector<std::vector<std::pair<bool, Eigen::Vector3d>>> thread_results(num_threads);  // Results per thread.

        // Determine the number of rays each thread will process.
        int batch_size = rays_.size() / num_threads;

        // Split the rays into batches for each thread.
        for (int i = 0; i < num_threads; ++i) {
            // Define the start and end of the batch for this thread
            auto begin = rays_.begin() + i * batch_size;
            auto end = (i == num_threads - 1) ? rays_.end() : begin + batch_size;

            // Reserve space in the result vector for this thread
            thread_results[i].reserve(std::distance(begin, end));

            // Launch a thread to process the assigned batch of rays
            threads.emplace_back([&, begin, end, i]() {
                for (auto it = begin; it != end; ++it) {
                    // Set up the ray origin (viewpoint position) and the ray direction
                    octomap::point3d ray_origin(position_.x(), position_.y(), position_.z());
                    octomap::point3d ray_end(it->x(), it->y(), it->z());

                    // Variable to store the hit point
                    octomap::point3d hit;
                    double ray_length = (ray_end - ray_origin).norm();  // Calculate the length of the ray
                    
                    // Perform the raycasting
                    bool is_hit = octomap->castRay(ray_origin, ray_end - ray_origin, hit, true, ray_length);

                    // If the ray hits an occupied voxel
                    if (is_hit) {
                        // Store the hit point in the thread's results
                        thread_results[i].emplace_back(true, Eigen::Vector3d(hit.x(), hit.y(), hit.z()));

                        // Retrieve the hit voxel
                        octomap::ColorOcTreeNode* hitNode = octomap->search(hit);

                        if (hitNode) {
                            // Get the current color of the hit voxel
                            octomap::ColorOcTreeNode::Color currentColor = hitNode->getColor();

                            // // Blend the current color with light green
                            // uint8_t new_r = currentColor.r;  // Keep the red channel as is
                            // uint8_t new_g = std::min<uint8_t>(currentColor.g + green_increment, 255);  // Increase the green channel but clamp it at 255
                            // uint8_t new_b = currentColor.b;  // Keep the blue channel as is

                            // If the voxel was never colored, initialize it to light green
                            // if (currentColor.r == 255 && currentColor.g == 255 && currentColor.b == 255) {
                            //     new_r = light_green_r;
                            //     new_g = light_green_g;
                            //     new_b = light_green_b;
                            // } 


                            // Set the new color back to the hit voxel
                            hitNode->setColor(0, 255, 0);
                        }
                    } else {
                        // No hit: store a zero vector in the thread's results
                        thread_results[i].emplace_back(false, Eigen::Vector3d::Zero());
                    }
                }
            });
        }

        // Wait for all threads to complete execution
        for (auto& thread : threads) {
            thread.join();
        }

        // Combine the results from all threads into the final hit_results vector
        hit_results.clear();
        for (const auto& result : thread_results) {
            hit_results.insert(hit_results.end(), result.begin(), result.end());
        }

    } else {
        // Sequential version: process the rays one by one without multithreading
        for (int i = 0; i < rays_.size(); ++i) {
            const auto& ray = rays_[i];  // Get the current ray
            octomap::point3d ray_origin(position_.x(), position_.y(), position_.z());
            octomap::point3d ray_end(ray.x(), ray.y(), ray.z());

            // Variable to store the hit point
            octomap::point3d hit;
            // Perform the raycasting
            bool is_hit = octomap->castRay(ray_origin, ray_end - ray_origin, hit, true, (ray_end - ray_origin).norm());

            // If the ray hits an occupied voxel
            if (is_hit) {
                hit_results[i] = std::make_pair(true, Eigen::Vector3d(hit.x(), hit.y(), hit.z()));

                // Retrieve the hit voxel
                octomap::ColorOcTreeNode* hitNode = octomap->search(hit);

                if (hitNode) {
                    // Get the current color of the hit voxel
                    octomap::ColorOcTreeNode::Color currentColor = hitNode->getColor();

                    // Blend the current color with light green
                    uint8_t new_r = currentColor.r;  // Keep the red channel as is
                    uint8_t new_g = std::min<uint8_t>(currentColor.g + green_increment, 255);  // Increase the green channel but clamp it at 255
                    uint8_t new_b = currentColor.b;  // Keep the blue channel as is

                    // If the voxel was never colored, initialize it to light green
                    if (currentColor.r == 0 && currentColor.g == 0 && currentColor.b == 0) {
                        new_r = light_green_r;
                        new_g = light_green_g;
                        new_b = light_green_b;
                    }

                    // Set the new color back to the hit voxel
                    hitNode->setColor(new_r, new_g, new_b);
                }
            } else {
                // No hit: store a zero vector in the result
                hit_results[i] = std::make_pair(false, Eigen::Vector3d::Zero());
            }
        }
    }

    // Return the vector containing the results of the raycasting
    return hit_results;
}