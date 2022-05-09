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
#include "rcl_interfaces/msg/set_parameters_result.hpp"


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
        
        	this->declare_parameter<float>("max_vel_target", 0.8);  
			this->max_vel_target = this->get_parameter("max_vel_target").get_value<float>();
			
			this->declare_parameter<string>("trajectory", "linear");
			this->trajectory = this->get_parameter("trajectory").get_value<string>();
			
			this->declare_parameter<float>("max_height", 3.5);
			this->max_height = this->get_parameter("max_height").get_value<float>();
			this->max_height -= 0.5;  // Buffer, avoid "out of area"
			
			this->declare_parameter<float>("min_height", 1.8);
			this->min_height = this->get_parameter("min_height").get_value<float>();
			
			this->declare_parameter<float>("max_side", 5.0); 
			this->max_side = this->get_parameter("max_side").get_value<float>();
			this->max_side -= 2;  // Buffer, avoid "out of area"
			
			this->declare_parameter<float>("min_vel_target", 0.4);
			this->min_vel_target = this->get_parameter("min_vel_target").get_value<float>();
			
			this->w_z = min_height + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_height-min_height)));
			this->w_y = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_side-0.0)));
			this->w_x = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_side-0.0)));
			this->target_w = min_vel_target + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_vel_target-min_vel_target)));  // Angular velocity
			this->target_vy = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_vel_target-0.0)));
			this->target_vx = min_vel_target + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_vel_target-min_vel_target)));
			
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
                // Get state
                this->GetState("irlock_beacon");
                // If listening to actions and target state available
                if (this->success_set_new_state && this->success_get_new_state && this->play==1 && this->reset==0 && this->micrortps_connected) {
                	this->success_set_new_state = false;
					// Set new state
					this->move_target_pos();
				}
            };

            auto timer_callback = [this]() -> void {
                if (this->offboard_setpoint_counter_ == 30 && this->micrortps_connected) {
		            // Change to Offboard mode after 30 setpoints
		            this->publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);

		            // Arm the vehicle
		            this->arm();
		            
		            // Get targetstate
		            this->GetState("irlock_beacon");
		            // If trajectory is circular initialize radius and angular velocity and position on circle
					if(this->success_get_new_state && this->trajectory.compare("circular")==0) {
						double theta = 2 * M_PI * (double)rand() / RAND_MAX;
						this->r = 1.0+ static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(3.0-1.0)));  // Minimum and maximum radius
						this->success_set_new_state = false;
						this->ir_beacon_pose.position.x = this->r * cos(theta);
						this->ir_beacon_pose.position.y = this->r * sin(theta);
						this->move_target_pos();  // Must be called after the new position has been set in this->ir_beacon_pose
					}
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
            timer_target_ = this->create_wall_timer(milliseconds(this->target_period), target_timer_callback);
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
        float eps_pos = 0.15;  // Tolerance for position, used in takeoff
        float eps_vel = 0.05;  // Tolerance for velocity, used in takeoff
        float vx, vy, vz = 0.0;
        float target_vx, target_vy, target_w;
        float max_vel_target;
        float w_vx, w_vy, w_vz = 0.0;
        float x, y, z, r = 0.0;
        float prev_x, prev_y, prev_z = 0.0;
        int play = 0;  // if 1 listen to velocity from agent, 0 stop publish velocity
        int reset = 1;  // if 1 perform takeoff, 0 takeoff complete
        int target_period = 10;  // Period for target timer
        string trajectory;  // String used to select target trajectory
        float circular_angle = 0.0;  // Used, if trajectory=='circular', to keep track of position in the circle
        bool velocity_reversed = false;  // flag used when checking the position of the target, avoid resetting multiple times
        float max_height, min_height, max_side, min_vel_target;
        float w_x, w_y, w_z;
        bool micrortps_connected = false;  // Whether micrortps_agent and gazebo are ready or not
        bool armed_flag = false;  // Drone armed
        geometry_msgs::msg::Pose ir_beacon_pose;
        geometry_msgs::msg::Twist ir_beacon_twist;
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
        void reset_target();
        void move_target_pos();
        void reset_target_velocity();
        void check_pos_target();
        void update_circular_angle();
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
        this->w_x = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_side-0.0)));
    } else {
        this->w_x = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_side-0.0)));
        this->w_x = -this->w_x;
    }
    sign = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    if(sign >= 0.5){
        this->w_y = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_side-0.0)));
    } else {
        this->w_y = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_side-0.0)));
        this->w_y = -this->w_y;
    }
    this->w_z = this->min_height + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_height-this->min_height)));
    
    // Resetting target
    this->reset_target();

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
  	this->ir_beacon_pose = future.get()->state.pose;
  	this->ir_beacon_twist = future.get()->state.twist;
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

// Set new position for target using a service, sets the velocity to zero
void EnvNode::move_target_pos() {
	geometry_msgs::msg::Point p = geometry_msgs::msg::Point();
	geometry_msgs::msg::Pose pose = geometry_msgs::msg::Pose();
	geometry_msgs::msg::Vector3 lin_vel = geometry_msgs::msg::Vector3();
	geometry_msgs::msg::Vector3 ang_vel = geometry_msgs::msg::Vector3();
	lin_vel = this->ir_beacon_twist.linear;
	ang_vel = this->ir_beacon_twist.angular;
	
	this->check_pos_target();  // If target is near border make it go backwards
	
	if(this->trajectory.compare("linear")==0){
		x = this->ir_beacon_pose.position.x + this->target_vx* this->target_period/1000;
		y = this->ir_beacon_pose.position.y + this->target_vy* this->target_period/1000;
		z = this->ir_beacon_pose.position.z;
	} else if(this->trajectory.compare("circular")==0){
	    // Using only of velocity as module
		x = this->r * cos(this->circular_angle);
		y = this->r * sin(this->circular_angle);
		z = this->ir_beacon_pose.position.z;
		this->update_circular_angle();
	}
	p.x = x; p.y = y; p.z = z;
	pose.position = p; pose.orientation = this->ir_beacon_pose.orientation;
	this->SetState("irlock_beacon", pose, lin_vel, ang_vel);
}

void EnvNode::reset_target(){
	// Resetting target velocity
	float sign, w_x, w_y;
    this->reset_target_velocity();  // Needs to be called before "move_target_pos"
    this->circular_angle = 0.0;  // Reset angle, position in circle
    
    // Sample random position for target
    if(this->trajectory.compare("linear")==0){
		sign = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		if(sign >= 0.5){
		    w_x = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_side/2-0.0)));
		} else {
		    w_x = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_side/2-0.0)));
		    w_x = -this->w_x;
		}
		sign = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		if(sign >= 0.5){
		    w_y = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_side/2-0.0)));
		} else {
		    w_y = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_side/2-0.0)));
		    w_y = -this->w_y;
		}
    } else if (this->trajectory.compare("circular")==0){
    	double theta = 2 * M_PI * (double)rand() / RAND_MAX;
    	this->r = 1.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(3.0-1.0)));  // Minimum and maximum radius
		w_x = this->r * cos(theta);
		w_y = this->r * sin(theta);
    }
    // Use service to set new position
    this->ir_beacon_pose.position.x = w_x;
    this->ir_beacon_pose.position.y = w_y;
    this->move_target_pos();  // Must be called after the new position has been set in this->ir_beacon_pose
}

void EnvNode::reset_target_velocity(){
	float sign;
	
	if(this->trajectory.compare("linear")==0) {
		this->target_vx = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_vel_target)));
		this->target_vy = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_vel_target)));
		while(this->target_vx<0.4 && this->target_vy<0.4) {  // Avoid having a slow target
		    this->target_vx = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_vel_target)));
			this->target_vy = 0.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_vel_target)));
		}
		sign = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		if(sign >= 0.5){
		    this->target_vy = -this->target_vy;
		}
		
		sign = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		if(sign >= 0.5){
			this->target_vx = -this->target_vx;
		}
    } else if(this->trajectory.compare("circular")==0) {
        this->target_w = 0.4 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->max_vel_target-0.4)));
        sign = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		if(sign >= 0.5){
			this->target_w = -this->target_w;
		}
    }
    
    
    this->velocity_reversed = false;
}

void EnvNode::check_pos_target(){
    // this->max_side is the maximum position in which the target can spawn, this->max_side+2 is the maximum position in which the drone can fly
	if(!this->velocity_reversed && (std::abs(this->ir_beacon_pose.position.x)>this->max_side+1 || std::abs(this->ir_beacon_pose.position.y)>this->max_side+1)){
		this->target_vx = -this->target_vx;
		this->target_vy = -this->target_vy;
		this->velocity_reversed = true;
	}
}

void EnvNode::update_circular_angle(){
	this->circular_angle += (this->target_w / this->r) * this->target_period/1000;
	if(this->circular_angle>=2*M_PI){
		this->circular_angle -= 2*M_PI;
	} else if(this->circular_angle<=-2*M_PI){
		this->circular_angle += 2*M_PI;
	}
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

