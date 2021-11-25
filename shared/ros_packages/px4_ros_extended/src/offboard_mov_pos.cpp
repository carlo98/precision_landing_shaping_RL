#include <iostream>
#include <stdint.h>
#include <chrono>
#include <cmath>

/* ROS2 libs */
#include <rclcpp/rclcpp.hpp>

/* PX4 libs */
#include <px4_msgs/msg/timesync.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_control_mode.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>




using namespace std;
using namespace std::chrono;
using namespace std::chrono_literals;
using namespace px4_msgs::msg;
using std::placeholders::_1;



// param set COM_RCL_EXCEPT 4

class OffboardControl : public rclcpp::Node {

    public:

        // Constructor
        OffboardControl() : Node("offboard_control_node") 
        {
            #ifdef ROS_DEFAULT_API
                offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("fmu/offboard_control_mode/in", 10);
                trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("fmu/trajectory_setpoint/in", 10);
                vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("fmu/vehicle_command/in", 10);
                vehicle_odometry_subscriber = this->create_subscription<VehicleOdometry>("fmu/vehicle_odometry/out", 10, std::bind(&OffboardControl::vehicle_odometry_callback, this, _1));
            #else
                offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("fmu/offboard_control_mode/in");
                trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("fmu/trajectory_setpoint/in");
                vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("fmu/vehicle_command/in");
                vehicle_odometry_subscriber = this->create_subscription<VehicleOdometry>("fmu/vehicle_odometry/out", std::bind(&OffboardControl::vehicle_odometry_callback, this, _1));
            #endif

            // Subscriber: Timestamp
            timesync_sub_ = this->create_subscription<Timesync>(
                "fmu/timesync/out", 10,
                [this](const Timesync::UniquePtr msg) 
                {
                    timestamp_.store(msg->timestamp);
                });


            // Main loop
            offboard_setpoint_counter_ = 0;
            flight_mode_flag = 0;
            cont = 0;

		    auto timer_callback = [this]() -> void {

                if (offboard_setpoint_counter_ == 10) {
                    // Change to Offboard mode after 10 setpoints
                    this->publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);

                    // Arm the vehicle
                    this->arm();
                    offboard_setpoint_counter_++;
                }
                else{
                    // offboard_control_mode needs to be paired with trajectory_setpoint
                    this->publish_offboard_control_mode(true, false, false, false, false);
                    publish_trajectory_setpoint_pos(0.0, 0.0, this->wanted_height, 3.14);
                    this->offboard_setpoint_counter_++;
                }
                if(offboard_setpoint_counter_ >= 10){
                    if(flight_mode_flag == 0){
                        this->takeoff_position(this->wanted_height);
                        RCLCPP_INFO(this->get_logger(), "Takeoff started.");
                        flight_mode_flag = 1;
                    }
                    else if(flight_mode_flag == 1){
                        if(cont>=1){
                            cont = 0;
                            flight_mode_flag = 2;
                            RCLCPP_INFO(this->get_logger(), "Takeoff finished.");
                            RCLCPP_INFO(this->get_logger(), "Movement started.");
                        }
                    }
                    else if(flight_mode_flag == 2){
                        if(cont >= 1){
                            cont = 0;
                            flight_mode_flag = 3;
                            RCLCPP_INFO(this->get_logger(), "Movement finished.");
                            RCLCPP_INFO(this->get_logger(), "Stopping");
                        }
                        else{
                            this->publish_offboard_control_mode(true, false, false, false, false);
                            this->publish_trajectory_setpoint_pos(this->wanted_x_mov, 0.0, this->wanted_height, 3.14);
                        }
                    }
                    else if(flight_mode_flag == 3){
                        this->publish_offboard_control_mode(true, false, false, false, false);
                        this->publish_trajectory_setpoint_pos(this->wanted_x_mov, 0.0, this->wanted_height, 3.14);
                    }
                }
            };

            timer_ = this->create_wall_timer(100ms, timer_callback);

        }

        void arm() const;

        void disarm() const;
        
        void takeoff() const;
        
        void takeoff_position(float altitude) const;


    private:

        rclcpp::TimerBase::SharedPtr timer_;

        rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
        rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
        rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
        rclcpp::Subscription<Timesync>::SharedPtr timesync_sub_;
        rclcpp::Subscription<VehicleOdometry>::SharedPtr vehicle_odometry_subscriber;

        std::atomic<uint64_t> timestamp_;

        uint64_t offboard_setpoint_counter_;   //!< counter for the number of setpoints sent
        uint64_t cont;  // Counter for position reached
        uint64_t flight_mode_flag;  // Internal flight mode: 0 for takeoff, 1 takeoff in progress, 2 for moving, 3 for stopped
        float wanted_height = 1.25;
        float wanted_x_mov = 1.0;
        float eps_pos = 0.2;

        void publish_offboard_control_mode(bool pos, bool vel, bool acc, bool att, bool br) const;
        void publish_trajectory_setpoint_pos(float x, float y, float z, float yaw) const;
        void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0) const;
        void vehicle_odometry_callback(const VehicleOdometry::SharedPtr msg);
        
    // protected:

};


// Arm
void OffboardControl::arm() const 
{
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);

	RCLCPP_INFO(this->get_logger(), "Arm command send");
}


// Disarm
void OffboardControl::disarm() const 
{
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0);

	RCLCPP_INFO(this->get_logger(), "Disarm command send");
}


// Takeoff
void OffboardControl::takeoff() const 
{
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_TAKEOFF);
}


// Takeoff position
void OffboardControl::takeoff_position(float altitude) const 
{
	this->publish_offboard_control_mode(true, false, false, false, false);
	this->publish_trajectory_setpoint_pos(0.0, 0.0, altitude, 3.14);
}


// Receive odometry
void OffboardControl::vehicle_odometry_callback(const VehicleOdometry::SharedPtr msg)
{
    if(flight_mode_flag == 1 && std::abs(std::abs(msg->z) - wanted_height) <= eps_pos) {
        cont += 1;
    }
    else if(flight_mode_flag == 2 && std::abs(std::abs(msg->x) - wanted_x_mov) <= eps_pos) {
        cont += 1;
    }
}


// Offboard control mode: Position, Velocity, Acceleration
void OffboardControl::publish_offboard_control_mode(bool pos, bool vel, bool acc, bool att, bool br) const 
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


// Control setpoint movement position
void OffboardControl::publish_trajectory_setpoint_pos(float x, float y, float z, float yaw) const 
{
	TrajectorySetpoint msg{};
	msg.timestamp = timestamp_.load();
	msg.x = - x;
	msg.y = - y;
	msg.z = - z;
	msg.yaw = yaw; // [-PI:PI]

	trajectory_setpoint_publisher_->publish(msg);
}


// Send commands
void OffboardControl::publish_vehicle_command(uint16_t command, float param1, float param2) const 
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



/* Main */
int main(int argc, char* argv[])
{
    cout << "Starting offboard node" << endl;
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);

    // Initialize ROS node
	rclcpp::init(argc, argv);

    // Spin ROS node
    rclcpp::spin(std::make_shared<OffboardControl>());

    // Shutdown ROS node
    rclcpp::shutdown();

    return 0;
}
