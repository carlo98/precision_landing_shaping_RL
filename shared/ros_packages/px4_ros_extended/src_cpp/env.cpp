#include <iostream>
#include <stdint.h>
#include <unistd.h>
#include <chrono>
#include <math.h>
#include <ctime>

/* custom message */
#include "custom_msgs/msg/float32_multi_array.hpp"

/* ROS2 libs */
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int64.hpp>

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
using namespace std_msgs::msg;
using namespace px4_msgs::msg;
using namespace custom_msgs::msg;
using std::placeholders::_1;


class EnvNode : public rclcpp::Node {

    public:
        EnvNode() : Node("env_node")
        {
            vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("fmu/vehicle_command/in", 2);
            offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("fmu/offboard_control_mode/in", 2);
            odometry_subscriber = this->create_subscription<VehicleOdometry>("fmu/vehicle_odometry/out", 1, std::bind(&EnvNode::odometry_callback, this, _1));
            trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("fmu/trajectory_setpoint/in", 2);
            agent_action_received_publisher_ = this->create_publisher<Int64>("/agent/action_received", 1);
            agent_subscriber = this->create_subscription<Float32MultiArray>("/agent/velocity", 1, std::bind(&EnvNode::agent_callback, this, _1));
            play_reset_subscriber = this->create_subscription<Float32MultiArray>("/env/play_reset", 1, std::bind(&EnvNode::play_reset_callback, this, _1));
            play_reset_publisher = this->create_publisher<Float32MultiArray>("/env/play_reset", 1);
            
            srand (static_cast <unsigned> (time(0)));

            timesync_sub_ = this->create_subscription<Timesync>(
                "fmu/timesync/out", 1,
                [this](const Timesync::UniquePtr msg) 
                {
                    timestamp_.store(msg->timestamp);
                });

            auto timer_callback = [this]() -> void {
            
                if (this->offboard_setpoint_counter_ == 30) {
		    // Change to Offboard mode after 30 setpoints
		    this->publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);

		    // Arm the vehicle
		    this->arm();
		    usleep(1000000);
                }
                
		if(this->play==1 && this->reset==0) {
                    this->land();
                }
                else if(this->play==0 && this->reset==1) {
                    this->takeoff(this->x, this->y, this->z);
                }
                
                if(this->offboard_setpoint_counter_ <= 30) {
                    this->takeoff(this->x, this->y, this->z);
                    this->offboard_setpoint_counter_ += 1;
                }
            };

            timer_ = this->create_wall_timer(33ms, timer_callback);
        }

        void disarm() const;
        void land() const;
        void arm() const;

    private:
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
        rclcpp::Subscription<Timesync>::SharedPtr timesync_sub_;
        rclcpp::Subscription<VehicleOdometry>::SharedPtr odometry_subscriber;
        rclcpp::Subscription<Float32MultiArray>::SharedPtr agent_subscriber;
        rclcpp::Publisher<Int64>::SharedPtr agent_action_received_publisher_;
        rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
        rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
        rclcpp::Publisher<Float32MultiArray>::SharedPtr play_reset_publisher;
        rclcpp::Subscription<Float32MultiArray>::SharedPtr play_reset_subscriber;

        std::atomic<uint64_t> timestamp_;
        Int64 int64Msg = Int64();
        Float32MultiArray float32Msg = Float32MultiArray();
        std::vector<float> float32Vector = std::vector<float>(3);
        int offboard_setpoint_counter_ = 0;
        float eps_pos = 0.3;  // Tolerance for position
        float eps_vel = 0.05;  // Tolerance for velocity
        float vz = 0.0;
        float vx = 0.0;
        float vy = 0.0;
        int play = 0;  // if 1 listen to velocity from agent, 0 stop publish velocity
        int reset = 1;  // if 1 perform takeoff, 0 takeoff complete
        float x = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(10.0-0.0)));
        float y = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(10.0-0.0)));
        float z = 1.5 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(6.0-1.5)));


        void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0) const;
        void publish_trajectory_setpoint_vel(float vx, float vy, float vz, float yawspeed) const;
        void takeoff(float x, float y, float z) const;
        void agent_callback(const Float32MultiArray::SharedPtr msg_float);
        void play_reset_callback(const Float32MultiArray::SharedPtr msg_float);
        void odometry_callback(const VehicleOdometry::SharedPtr msg);
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
    
    this->int64Msg.data = 1;
    this->agent_action_received_publisher_->publish(this->int64Msg);
}

// Receive play_reset signal from agent
void EnvNode::play_reset_callback(const Float32MultiArray::SharedPtr msg_float)
{
    this->play = msg_float->data[0];  // if 1 listen to velocity from agent, 0 stop publish velocity
    
    if(this->reset == 0 && msg_float->data[1]==1) {  // Restart takeoff, sample random position
    
        float sign = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        if(sign >= 0.5){
            this->x = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(10.0-0.0)));
        } else {
            this->x = - 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(10.0-0.0)));
        }
        sign = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        if(sign >= 0.5){
            this->y = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(10.0-0.0)));
        } else {
            this->y = - 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(10.0-0.0)));
        }
        this->z = 1.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(5.0-1.0)));
    }
    
    this->reset = msg_float->data[1];  // if 1 perform takeoff, 0 takeoff complete
}

// Check successful takeoff
void EnvNode::odometry_callback(const VehicleOdometry::SharedPtr msg)
{
    if(std::abs((std::abs(this->x)-std::abs(msg->x))) <= this->eps_pos &&
       std::abs((std::abs(this->y)-std::abs(msg->y))) <= this->eps_pos && 
       std::abs((std::abs(this->z)-std::abs(msg->z))) <= this->eps_pos &&
       this->reset == 1){

         this->int64Msg.data = 1;
         this->agent_action_received_publisher_->publish(this->int64Msg);
         
         this->float32Vector.clear();
         this->float32Vector.push_back(1);  // Play
         this->float32Vector.push_back(0);  // Reset

         this->float32Msg.data = this->float32Vector;
         this->play_reset_publisher->publish(this->float32Msg);
    }
}

void EnvNode::disarm() const
{
    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND, 0.0);
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0);

	RCLCPP_INFO(this->get_logger(), "Disarm command send");
}

void EnvNode::arm() const {
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);

	RCLCPP_INFO(this->get_logger(), "Arm command send");
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