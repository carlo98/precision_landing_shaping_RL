#include <iostream>
#include <stdint.h>
#include <unistd.h>
#include <chrono>
#include <math.h>
#include <ctime>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int64.hpp>
#include "std_msgs/msg/float32_multi_array.hpp"

#include <px4_msgs/msg/timesync.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_control_mode.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <gazebo_msgs/srv/get_entity_state.hpp>
#include <gazebo_msgs/srv/set_entity_state.hpp>


using namespace std;
using namespace std::chrono;
using namespace std::chrono_literals;
using namespace std_msgs::msg;
using namespace px4_msgs::msg;
using namespace gazebo_msgs::msg;
using std::placeholders::_1;


class EnvNode : public rclcpp::Node {

    public:
        EnvNode() : Node("env_node")
        {
            vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("fmu/vehicle_command/in", 2);
            offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("fmu/offboard_control_mode/in", 2);
            odometry_subscriber = this->create_subscription<VehicleOdometry>("fmu/vehicle_odometry/out", 1, std::bind(&EnvNode::odometry_callback, this, _1));
            trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("fmu/trajectory_setpoint/in", 2);
            status_subscriber = this->create_subscription<VehicleStatus>("fmu/vehicle_status/out", 1, std::bind(&EnvNode::status_callback, this, _1));
            agent_action_received_publisher_ = this->create_publisher<Int64>("/agent/action_received", 1);
            agent_subscriber = this->create_subscription<Float32MultiArray>("/agent/velocity", 1, std::bind(&EnvNode::agent_callback, this, _1));
            play_reset_subscriber = this->create_subscription<Float32MultiArray>("/env/play_reset/in", 1, std::bind(&EnvNode::play_reset_callback, this, _1));
            play_reset_publisher = this->create_publisher<Float32MultiArray>("/env/play_reset/out", 1);
            agent_odom_publisher = this->create_publisher<Float32MultiArray>("/agent/odom", 1);
            resetting_subscriber = this->create_subscription<Int64>("/env/resetting", 1, std::bind(&EnvNode::resetting_callback, this, _1));
            get_state_client_ = this->create_client<gazebo_msgs::srv::GetEntityState>("/gazebo/get_entity_state");
			get_state_client_->wait_for_service(std::chrono::seconds(1));
			set_state_client_ = this->create_client<gazebo_msgs::srv::SetEntityState>("/gazebo/set_entity_state");
			set_state_client_->wait_for_service(std::chrono::seconds(1));
            
            srand (static_cast <unsigned> (time(0)));

            timesync_sub_ = this->create_subscription<Timesync>(
                "fmu/timesync/out", 1,
                [this](const Timesync::UniquePtr msg) 
                {
                    timestamp_.store(msg->timestamp);
                });
                
            // Target time callback
            auto target_timer_callback = [this]() -> void {
                // Get initial state
                this->GetState("irlock_beacon");
                if (this->success_set_new_state && this->success_get_new_state) {
                	this->success_set_new_state = false;
					// Set new state
					geometry_msgs::msg::Point p = geometry_msgs::msg::Point();
					geometry_msgs::msg::Pose pose = geometry_msgs::msg::Pose();
					geometry_msgs::msg::Vector3 lin_vel = geometry_msgs::msg::Vector3();
					geometry_msgs::msg::Vector3 ang_vel = geometry_msgs::msg::Vector3();
					p.x = this->new_ir_beacon_pose.position.x+0.01; p.y = this->new_ir_beacon_pose.position.y; p.z = this->new_ir_beacon_pose.position.z;
					pose.position = p; pose.orientation = this->new_ir_beacon_pose.orientation;
					lin_vel = this->new_ir_beacon_twist.linear;
					ang_vel = this->new_ir_beacon_twist.angular;
					this->SetState("irlock_beacon", pose, lin_vel, ang_vel);
				}
            };

            auto timer_callback = [this]() -> void {
                if (this->offboard_setpoint_counter_ == 30 && this->micrortps_connected) {
		            // Change to Offboard mode after 30 setpoints
		            this->publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);

		            // Arm the vehicle
		            this->arm();
		            usleep(1000000);
                }

		        if(this->play==1 && this->reset==0 && this->micrortps_connected) {  // Listen to actions
		            this->agent_odom_pub();
                    this->land();
                }
                else if(this->play==0 && this->reset==1 && this->micrortps_connected) {  // Go to new position or hover (avoids failsafe activation)
                    if(this->offboard_setpoint_counter_ <= 30) {
                        this->offboard_setpoint_counter_ += 1;
                    }
                    else if(this->offboard_setpoint_counter_ > 30 && !this->armed_flag){  // Repeat shorter arming procedure
                        this->offboard_setpoint_counter_ = 15;
                    }
                    this->takeoff(this->w_x, this->w_y, this->w_z);
                }
            };
            timer_target_ = this->create_wall_timer(20ms, target_timer_callback);  // 50 Hz
            timer_ = this->create_wall_timer(50ms, timer_callback);  // 20Hz
        }

        void arm() const;
        void disarm() const;
        void land() const;

    private:
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::TimerBase::SharedPtr timer_target_;
        rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
        rclcpp::Subscription<Timesync>::SharedPtr timesync_sub_;
        rclcpp::Subscription<VehicleOdometry>::SharedPtr odometry_subscriber;
        rclcpp::Subscription<Float32MultiArray>::SharedPtr agent_subscriber;
        rclcpp::Subscription<VehicleStatus>::SharedPtr status_subscriber;
        rclcpp::Publisher<Int64>::SharedPtr agent_action_received_publisher_;
        rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
        rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
        rclcpp::Publisher<Float32MultiArray>::SharedPtr play_reset_publisher;
        rclcpp::Subscription<Float32MultiArray>::SharedPtr play_reset_subscriber;
        rclcpp::Publisher<Float32MultiArray>::SharedPtr agent_odom_publisher;
        rclcpp::Subscription<Int64>::SharedPtr resetting_subscriber;
        std::shared_ptr<rclcpp::Client<gazebo_msgs::srv::GetEntityState>> get_state_client_;
  		std::shared_ptr<rclcpp::Client<gazebo_msgs::srv::SetEntityState>> set_state_client_;

        std::atomic<uint64_t> timestamp_;
        Int64 int64Msg = Int64();
        Float32MultiArray float32Msg = Float32MultiArray();
        std::vector<float> float32Vector = std::vector<float>(3);
        int offboard_setpoint_counter_ = 0;
        float eps_pos = 0.15;  // Tolerance for position
        float eps_vel = 0.05;  // Tolerance for velocity
        float vx, vy, vz = 0.0;
        float w_vx, w_vy, w_vz = 0.0;
        float x, y, z = 0.0;
        float prev_x, prev_y, prev_z = 0.0;
        int play = 0;  // if 1 listen to velocity from agent, 0 stop publish velocity
        int reset = 1;  // if 1 perform takeoff, 0 takeoff complete
        int max_z = 3.0;
        int min_z = 1.8;
        int max_xy = 3.0;
        float w_x = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_xy-0.0)));
        float w_y = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_xy-0.0)));
        float w_z = min_z + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_z-min_z)));
        bool micrortps_connected = false;  // Whether micrortps_agent and gazebo are ready or not
        bool armed_flag = false;  // Drone armed
        geometry_msgs::msg::Pose new_ir_beacon_pose;
        geometry_msgs::msg::Twist new_ir_beacon_twist;
        bool success_set_new_state = true;
        bool success_get_new_state = false;

        void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0) const;
        void publish_trajectory_setpoint_vel(float vx, float vy, float vz, float yawspeed) const;
        void takeoff(float x, float y, float z) const;
        void agent_callback(const Float32MultiArray::SharedPtr msg_float);
        void play_reset_callback(const Float32MultiArray::SharedPtr msg_float);
        void odometry_callback(const VehicleOdometry::SharedPtr msg);
        void resetting_callback(const Int64::SharedPtr msg);
        void status_callback(const VehicleStatus::SharedPtr msg);
        void agent_odom_pub();
        void publish_offboard_control_mode(bool pos, bool vel, bool acc, bool att, bool br) const;
        void new_position();
        void GetState(const std::string & _entity);
        void SetState(const std::string & _entity, const geometry_msgs::msg::Pose & _pose, 
        			  const geometry_msgs::msg::Vector3 & _lin_vel, const geometry_msgs::msg::Vector3 & _ang_vel);
};

void EnvNode::land() const
{
    this->publish_offboard_control_mode(false, true, false, false, false);
	this->publish_trajectory_setpoint_vel(this->w_vx, this->w_vy, this->w_vz, 0.0);
}

void EnvNode::status_callback(const VehicleStatus::SharedPtr msg)
{
    this->armed_flag = msg->arming_state == 2;  // If 2 drone is armed
    if(this->armed_flag){
        this->offboard_setpoint_counter_ = 31;
    }
}

// Receive action from agent
void EnvNode::agent_callback(const Float32MultiArray::SharedPtr msg_float)
{
    this->w_vx = msg_float->data[0];
    this->w_vy = msg_float->data[1];
    this->w_vz = msg_float->data[2];
    
    this->int64Msg.data = 1;
    this->agent_action_received_publisher_->publish(this->int64Msg);
}

void EnvNode::agent_odom_pub()
{
    this->float32Vector.clear();
    this->float32Vector.push_back(this->x);
    this->float32Vector.push_back(this->y);
    this->float32Vector.push_back(this->z);
    this->float32Vector.push_back(this->vx);
    this->float32Vector.push_back(this->vy);
    this->float32Vector.push_back(this->vz);

    this->float32Msg.data = this->float32Vector;
    this->agent_odom_publisher->publish(this->float32Msg);
}

// Receive play_reset signal from agent
void EnvNode::play_reset_callback(const Float32MultiArray::SharedPtr msg_float)
{
    if(this->play==0 && msg_float->data[0]==1 && msg_float->data[1]==0){
        this->agent_odom_pub();
    }
    this->play = msg_float->data[0];  // if 1 listen to velocity from agent, 0 stop publish velocity
    this->reset = msg_float->data[1];  // if 1 perform takeoff, 0 takeoff complete

    if(this->play==1 && this->reset==1){
        // Reset velocities
        this->w_vx = 0.0;
        this->w_vy = 0.0;
        this->w_vz = 0.0;
        this->new_position();
    }
}

void EnvNode::resetting_callback(const Int64::SharedPtr msg)
{
    if(this->micrortps_connected && msg->data == 1){
        this->offboard_setpoint_counter_ = 0;
        this->new_position();
    }
    this->micrortps_connected = msg->data == 0 && (this->x != 0.0 ||  this->y != 0.0 || this->z != 0.0);
}

void EnvNode::new_position(){

    // Sample random position
    float sign = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    if(sign >= 0.5){
        this->w_x = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_xy-0.0)));
    } else {
        this->w_x = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_xy-0.0)));
        this->w_x = -this->w_x;
    }
    sign = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    if(sign >= 0.5){
        this->w_y = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_xy-0.0)));
    } else {
        this->w_y = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_xy-0.0)));
        this->w_x = -this->w_y;
    }
    this->w_z = this->min_z + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_z-this->min_z)));

    this->reset = 1;
    this->play = 0;
}

// Check successful takeoff
void EnvNode::odometry_callback(const VehicleOdometry::SharedPtr msg)
{
    if(std::abs((std::abs(this->w_x)-std::abs(msg->x))) <= this->eps_pos &&
              std::abs((std::abs(this->w_y)-std::abs(msg->y))) <= this->eps_pos &&
                std::abs((std::abs(this->w_z)-std::abs(msg->z))) <= this->eps_pos &&
                  this->reset == 1 && this->play == 0){  // Arrived at initial position, training can start

         this->float32Vector.clear();
         this->float32Vector.push_back(0);  // Play, do not care, just listen to this
         this->float32Vector.push_back(0);  // Reset
         
         this->reset = 0;
         this->play = 1;
         this->w_vx = 0.0;
         this->w_vy = 0.0;
         this->w_vz = 0.0;

         this->float32Msg.data = this->float32Vector;
         this->play_reset_publisher->publish(this->float32Msg);
    }
    this->x = msg->x;
    this->y = msg->y;
    this->z = msg->z;
    this->vx = msg->vx;
    this->vy = msg->vy;
    this->vz = msg->vz;
}

void EnvNode::disarm() const
{
    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND, 0.0);
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0);

	RCLCPP_INFO(this->get_logger(), "Disarm command sent");
}

void EnvNode::arm() const {
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);

	RCLCPP_INFO(this->get_logger(), "Arm command sent");
}

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

// Helper function to call get state service
void EnvNode::GetState(const std::string & _entity) {
  auto request = std::make_shared<gazebo_msgs::srv::GetEntityState::Request>();
  request->name = _entity;

  auto response_received_callback = [this](rclcpp::Client<gazebo_msgs::srv::GetEntityState>::SharedFuture future) {
  	this->new_ir_beacon_pose = future.get()->state.pose;
  	this->new_ir_beacon_twist = future.get()->state.twist;
  	cout<<future.get()->success<<endl;
  	cout<<"Pose: ("<<"x="<<this->new_ir_beacon_pose.position.x<<", y="<<this->new_ir_beacon_pose.position.y<<", z="<<this->new_ir_beacon_pose.position.z<<")"<<endl;
  	cout<<"Velocity: ("<<"x="<<this->new_ir_beacon_twist.linear.x<<", y="<<this->new_ir_beacon_twist.linear.y<<", z="<<this->new_ir_beacon_twist.linear.z<<")"<<endl;
  	this->success_get_new_state = future.get()->success;
  };
  auto response_future = get_state_client_->async_send_request(request, response_received_callback);
}

// Helper function to call set state service
void EnvNode::SetState(const std::string & _entity, 
									const geometry_msgs::msg::Pose & _pose, 
									const geometry_msgs::msg::Vector3 & _lin_vel, 
									const geometry_msgs::msg::Vector3 & _ang_vel) {
  auto request = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
  request->state.name = _entity;
  request->state.pose.position = _pose.position;
  request->state.pose.orientation = _pose.orientation;
  request->state.twist.linear = _lin_vel;
  request->state.twist.angular = _ang_vel;
  
  auto response_received_callback = [this](rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedFuture future) {
  	this->success_set_new_state = future.get()->success;
  };

  auto response_future = set_state_client_->async_send_request(request, response_received_callback);
}

int main(int argc, char* argv[])
{
    cout << "Starting env node" << endl;
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EnvNode>());
    rclcpp::shutdown();

    return 0;
}

