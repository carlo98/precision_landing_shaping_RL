#include <iostream>
#include <stdint.h>
#include <unistd.h>
#include <chrono>
#include <math.h>

/* custom message */
#include "custom_msgs/msg/float32_multi_array.hpp"

/* ROS2 libs */
#include <rclcpp/rclcpp.hpp>

/* PX4 libs */
#include <px4_msgs/msg/timesync.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_control_mode.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>


using namespace std;
using namespace std::chrono;
using namespace std::chrono_literals;
using namespace px4_msgs::msg;
using namespace custom_msgs::msg;
using std::placeholders::_1;


class LandNode : public rclcpp::Node {

    public:

        // Constructor
        LandNode() : Node("land_node") 
        {
                vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("fmu/vehicle_command/in", 2);
                offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("fmu/offboard_control_mode/in", 2);
                vehicle_land_detected = this->create_subscription<VehicleOdometry>("fmu/vehicle_odometry/out", 2, std::bind(&LandNode::vehicle_land_detected_callback, this, _1));
                trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("fmu/trajectory_setpoint/in", 2);
                agent_subscriber = this->create_subscription<Float32MultiArray>("/agent/velocity", 2, std::bind(&LandNode::agent_callback, this, _1));

            // Subscriber: Timestamp
            timesync_sub_ = this->create_subscription<Timesync>(
                "fmu/timesync/out", 1,
                [this](const Timesync::UniquePtr msg) 
                {
                    timestamp_.store(msg->timestamp);
                });

            this->cont_my = 0;
            auto timer_callback = [this]() -> void {

               RCLCPP_INFO(this->get_logger(), "cont: %d", this->cont_my);
		if(this->cont_my < 50) {
                    this->land();
                }
                else {
                    this->disarm();
                    rclcpp::shutdown();
                }
            };

            timer_ = this->create_wall_timer(33ms, timer_callback);

        }

        void disarm() const;
        void land() const;

    private:

        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
        rclcpp::Subscription<Timesync>::SharedPtr timesync_sub_;
        rclcpp::Subscription<VehicleOdometry>::SharedPtr vehicle_land_detected;
        rclcpp::Subscription<Float32MultiArray>::SharedPtr agent_subscriber;
        rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
        rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
            

        std::atomic<uint64_t> timestamp_;
        int cont_my = 0;
        float eps_pos = 0.1;  // Tolerance for position
        float eps_vel = 0.05;  // Tolerance for velocity
        float vz = 0.0;
        float vx = 0.0;
        float vy = 0.0;


        void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0) const;
        void publish_trajectory_setpoint_vel(float vx, float vy, float vz, float yawspeed) const;
        void agent_callback(const Float32MultiArray::SharedPtr msg_float);
        void vehicle_land_detected_callback(const VehicleOdometry::SharedPtr msg);
        void publish_offboard_control_mode(bool pos, bool vel, bool acc, bool att, bool br) const;
};

// Land
void LandNode::land() const 
{
        this->publish_offboard_control_mode(false, true, false, false, false);
	this->publish_trajectory_setpoint_vel(this->vx, this->vy, this->vz, 0.0);
	RCLCPP_INFO(this->get_logger(), "Land command send");
}

// Receive action from agent
void LandNode::agent_callback(const Float32MultiArray::SharedPtr msg_float)
{
    this->vx = msg_float->data[0];
    this->vy = msg_float->data[1];
    this->vz = msg_float->data[2];
}

// Receive land detected
void LandNode::vehicle_land_detected_callback(const VehicleOdometry::SharedPtr msg)
{
    if(std::abs(msg->z) <= eps_pos && std::abs(msg->vz) <= eps_vel && std::abs(msg->vx) <= eps_vel) {
        this->cont_my += 1;
    }
}

// Disarm
void LandNode::disarm() const 
{
        publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND, 0.0);
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0);

	RCLCPP_INFO(this->get_logger(), "Disarm command send");
}

// Control setpoint movement velocity
void LandNode::publish_trajectory_setpoint_vel(float vx, float vy, float vz, float yawspeed) const 
{
    TrajectorySetpoint msg{};
    msg.timestamp = timestamp_.load();
    msg.x = NAN;
    msg.y = NAN;
    msg.z = NAN;
    msg.yaw = 3.14;
    msg.vx = - vx;
    msg.vy = - vy;
    msg.vz = - vz;
    msg.yawspeed = yawspeed; // [-PI:PI]
    
    trajectory_setpoint_publisher_->publish(msg);
}

// Send commands
void LandNode::publish_vehicle_command(uint16_t command, float param1, float param2) const 
{
	VehicleCommand msg{};
	msg.timestamp = timestamp_.load();
	msg.param1 = param1;
	msg.param2 = param2;
	msg.command = command;
	msg.target_system = 1;
	msg.target_component = 1;
	msg.source_system = 1;
	msg.source_component = 1;
	msg.from_external = true;

	vehicle_command_publisher_->publish(msg);
}

// Offboard control mode: Position, Velocity, Acceleration
void LandNode::publish_offboard_control_mode(bool pos, bool vel, bool acc, bool att, bool br) const 
{
	OffboardControlMode msg{};
	msg.timestamp = timestamp_.load();
	msg.position = pos;
	msg.velocity = vel;
	msg.acceleration = acc;
	msg.attitude = att;
	msg.body_rate = br;

	offboard_control_mode_publisher_->publish(msg);
}


/* Main */
int main(int argc, char* argv[])
{
    cout << "Starting land & disarm node" << endl;
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);

    // Initialize ROS node
    rclcpp::init(argc, argv);

    // Spin ROS node
    rclcpp::spin(std::make_shared<LandNode>());

    // Shutdown ROS node
    rclcpp::shutdown();

    return 0;
}
