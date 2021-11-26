#include <iostream>
#include <stdint.h>
#include <unistd.h>
#include <chrono>
#include <random>
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


class EnvNode : public rclcpp::Node {

    public:
        EnvNode() : Node("env_node")
        {
            vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("fmu/vehicle_command/in", 2);
            offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("fmu/offboard_control_mode/in", 2);
            vehicle_land_detected = this->create_subscription<VehicleOdometry>("fmu/vehicle_odometry/out", 2, std::bind(&EnvNode::vehicle_land_detected_callback, this, _1));
            trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("fmu/trajectory_setpoint/in", 2);
            agent_subscriber = this->create_subscription<Float32MultiArray>("/agent/velocity", 2, std::bind(&EnvNode::agent_callback, this, _1));
            agent_reward_publisher = this->create_subscription<Float32MultiArray>("/agent/reward", 2);
            play_reset_subscriber = this->create_subscription<Float32MultiArray>("/env/play_reset", 2, std::bind(&EnvNode::play_reset_callback, this, _1));

            timesync_sub_ = this->create_subscription<Timesync>(
                "fmu/timesync/out", 1,
                [this](const Timesync::UniquePtr msg) 
                {
                    timestamp_.store(msg->timestamp);
                });

            this->cont_my = 0;
            auto timer_callback = [this]() -> void {

               RCLCPP_INFO(this->get_logger(), "cont: %d", this->cont_my);
		        if(this->cont_my < 50 && this->play==1) {
                    this->land();
                }
                else if(this->play==0 && this->reset==1) {
                    this->takeoff(this->x, this->y, this->z);
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
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-10.0, 10.0);
        int cont_my = 0;
        float eps_pos = 0.1;  // Tolerance for position
        float eps_vel = 0.05;  // Tolerance for velocity
        float vz = 0.0;
        float vx = 0.0;
        float vy = 0.0;
        int play = 0;  // if 1 listen to velocity from agent, 0 stop publish velocity
        int reset = 1;  // if 1 perform takeoff, 0 takeoff complete
        float x = this->dis(this->gen);
        float y = this->dis(this->gen);
        float z = this->dis(this->gen);


        void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0) const;
        void publish_trajectory_setpoint_vel(float vx, float vy, float vz, float yawspeed) const;
        void takeoff(float x, float y, float z) const;
        void agent_callback(const Float32MultiArray::SharedPtr msg_float);
        void play_reset_callback(const Float32MultiArray::SharedPtr msg_float);
        void vehicle_land_detected_callback(const VehicleOdometry::SharedPtr msg);
        void publish_offboard_control_mode(bool pos, bool vel, bool acc, bool att, bool br) const;
};

// Land
void EnvNode::land() const
{
    this->publish_offboard_control_mode(false, true, false, false, false);
	this->publish_trajectory_setpoint_vel(this->vx, this->vy, this->vz, 0.0);
	RCLCPP_INFO(this->get_logger(), "Land command send");
}

// Receive action from agent
void EnvNode::agent_callback(const Float32MultiArray::SharedPtr msg_float)
{
    this->vx = msg_float->data[0];
    this->vy = msg_float->data[1];
    this->vz = msg_float->data[2];
}

// Receive play_reset signal from agent
void EnvNode::play_reset_callback(const Float32MultiArray::SharedPtr msg_float)
{
    this->play = msg_float->data[0];  // if 1 listen to velocity from agent, 0 stop publish velocity
    if(this->reset == 0 && msg_float->data[1]==1) {
        this->x = this->dis(this->gen);
        this->y = this->dis(this->gen);
        this->z = this->dis(this->gen);
    }
    this->reset = msg_float->data[1];  // if 1 perform takeoff, 0 takeoff complete
}

// Receive land detected
void EnvNode::vehicle_land_detected_callback(const VehicleOdometry::SharedPtr msg)
{
    if(std::abs(msg->z) <= eps_pos && std::abs(msg->vz) <= eps_vel && std::abs(msg->vx) <= eps_vel) {
        this->cont_my += 1;
    }
}

// Disarm
void EnvNode::disarm() const
{
    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND, 0.0);
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0);

	RCLCPP_INFO(this->get_logger(), "Disarm command send");
}

// Control setpoint movement velocity
void EnvNode::publish_trajectory_setpoint_vel(float vx, float vy, float vz, float yawspeed) const
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

// Control setpoint movement position
void EnvNode::takeoff(float x, float y, float z) const {
    this->publish_offboard_control_mode(true, false, false, false, false);
	TrajectorySetpoint msg{};
	msg.timestamp = timestamp_.load();
	msg.x = - x;
	msg.y = - y;
	msg.z = - z;
	msg.yaw = 3.14;

	trajectory_setpoint_publisher_->publish(msg);
}

// Send commands
void EnvNode::publish_vehicle_command(uint16_t command, float param1, float param2) const
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
void EnvNode::publish_offboard_control_mode(bool pos, bool vel, bool acc, bool att, bool br) const
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
    cout << "Starting env node" << endl;
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);

    // Initialize ROS node
    rclcpp::init(argc, argv);

    // Spin ROS node
    rclcpp::spin(std::make_shared<EnvNode>());

    // Shutdown ROS node
    rclcpp::shutdown();

    return 0;
}
