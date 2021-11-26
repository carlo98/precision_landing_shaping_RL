#include <iostream>
#include <stdint.h>
#include <chrono>
#include <math.h>

#include <rclcpp/rclcpp.hpp>

#include "custom_msgs/msg/float32_multi_array.hpp"
#include <px4_msgs/msg/timesync.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>


using namespace std;
using namespace std::chrono;
using namespace std::chrono_literals;
using namespace px4_msgs::msg;
using namespace custom_msgs::msg;
using std::placeholders::_1;


class BaselinePrecLandNode : public rclcpp::Node {

    public:

        // Constructor
        BaselinePrecLandNode() : Node("baseline_prec_land_node") 
        {
            pub_agent_vec = this->create_publisher<Float32MultiArray>("/agent/velocity", 2);
            vehicle_odometry_subscriber = this->create_subscription<VehicleOdometry>("fmu/vehicle_odometry/out", 2, std::bind(&BaselinePrecLandNode::vehicle_odometry_callback, this, _1));
            
            // Subscriber: Timestamp
            timesync_sub_ = this->create_subscription<Timesync>(
                "fmu/timesync/out", 2,
                [this](const Timesync::UniquePtr msg) 
                {
                    timestamp_.store(msg->timestamp);
                });


            // Main loop

            auto timer_callback = [this]() -> void {
                this->float32Vector.clear();
                this->float32Vector.push_back(0.2 * (this->x_drone - this->x_pos));
                this->float32Vector.push_back(0.2 * (this->y_drone - this->y_pos));
                this->float32Vector.push_back(0.5 * (this->z_drone - this->z_pos));

                this->float32Msg.data = this->float32Vector;
                this->pub_agent_vec->publish(this->float32Msg);
            };

            timer_ = this->create_wall_timer(200ms, timer_callback);

        }

    private:

        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<Float32MultiArray>::SharedPtr pub_agent_vec;
        rclcpp::Subscription<Timesync>::SharedPtr timesync_sub_;
        rclcpp::Subscription<VehicleOdometry>::SharedPtr vehicle_odometry_subscriber;
        void vehicle_odometry_callback(const VehicleOdometry::SharedPtr msg);
        
        std::atomic<uint64_t> timestamp_;
        
        float x_pos = 0.5;
        float y_pos = 1.0;
        float z_pos = 0.0;
        float x_drone = 0.0;
        float y_drone = 0.0;
        float z_drone = 0.0;
        Float32MultiArray float32Msg = Float32MultiArray();
        std::vector<float> float32Vector = std::vector<float>(3);
        
    // protected:

};

// Receive odometry
void BaselinePrecLandNode::vehicle_odometry_callback(const VehicleOdometry::SharedPtr msg)
{
    this->z_drone = msg->z;
    this->x_drone = msg->x;
    this->y_drone = msg->y;
}


/* Main */
int main(int argc, char* argv[])
{
    cout << "Starting baseline precision land node" << endl;
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);

    // Initialize ROS node
	rclcpp::init(argc, argv);

    // Spin ROS node
    rclcpp::spin(std::make_shared<BaselinePrecLandNode>());

    // Shutdown ROS node
    rclcpp::shutdown();

    return 0;
}
