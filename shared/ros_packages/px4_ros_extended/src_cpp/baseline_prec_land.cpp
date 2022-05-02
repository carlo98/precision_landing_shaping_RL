#include <iostream>
#include <stdint.h>
#include <chrono>
#include <math.h>

#include <rclcpp/rclcpp.hpp>

#include "std_msgs/msg/float32_multi_array.hpp"
#include <px4_msgs/msg/timesync.hpp>

#include "gazebo_msgs/msg/contacts_state.hpp"
#include <gazebo_msgs/srv/get_entity_state.hpp>
#include <gazebo_msgs/srv/set_entity_state.hpp>


using namespace std;
using namespace std::chrono;
using namespace std::chrono_literals;
using namespace px4_msgs::msg;
using namespace std_msgs::msg;
using namespace gazebo_msgs::msg;
using std::placeholders::_1;


class BaselinePrecLandNode : public rclcpp::Node {

    public:
        BaselinePrecLandNode() : Node("baseline_prec_land_node") 
        {
            bumper_subscriber = this->create_subscription<ContactsState>("/bumper_iris", rclcpp::SensorDataQoS(), std::bind(&BaselinePrecLandNode::check_landed_callback, this, _1));
            pub_agent_vec = this->create_publisher<Float32MultiArray>("/agent/velocity", 1);
            play_reset_publisher = this->create_publisher<Float32MultiArray>("/env/play_reset/in", 1);
            play_reset_subscriber = this->create_subscription<Float32MultiArray>("/env/play_reset/out", 1, std::bind(&BaselinePrecLandNode::play_reset_callback, this, _1));
            vehicle_odometry_subscriber = this->create_subscription<Float32MultiArray>("/agent/odom", 1, std::bind(&BaselinePrecLandNode::vehicle_odometry_callback, this, _1));
            
            get_state_client_ = this->create_client<gazebo_msgs::srv::GetEntityState>("/gazebo/get_entity_state");
			get_state_client_->wait_for_service(std::chrono::seconds(1));
			

			set_state_client_ = this->create_client<gazebo_msgs::srv::SetEntityState>("/gazebo/set_entity_state");
			set_state_client_->wait_for_service(std::chrono::seconds(1));

            timesync_sub_ = this->create_subscription<Timesync>(
                "fmu/timesync/out", 2,
                [this](const Timesync::UniquePtr msg) 
                {
                    timestamp_.store(msg->timestamp);
                });

            // Main loop
            auto timer_callback = [this]() -> void {
                // Get initial state
                this->GetState("irlock_beacon");
                if(this->reset == 0 and !this->stopped) {  // Playing, if 1 takeoff
		            landed_received = false;
		            
		            if (!this->landed) {
				        this->float32Vector.clear();
				        this->float32Vector.push_back(0.8 * (this->x_drone - this->x_pos));
				        this->float32Vector.push_back(0.8 * (this->y_drone - this->y_pos));
				        this->float32Vector.push_back(-0.3);

				        this->float32Msg.data = this->float32Vector;
				        cout << "Pos (x="<<(this->x_drone - this->x_pos)<<", y="<<(this->y_drone - this->y_pos)<<", z="<<this->z_drone<<")"<< endl;
				        cout << "Vel (x="<<0.8 * (this->x_drone - this->x_pos)<<", y="<<0.8 * (this->y_drone - this->y_pos)<<", z="<<-0.3<<")"<< endl;
				        this->pub_agent_vec->publish(this->float32Msg);
				        
				        if (this->success_set_new_state && this->success_get_new_state) {
				        	this->success_set_new_state = false;
							// Set new state
							geometry_msgs::msg::Point p = geometry_msgs::msg::Point();
							geometry_msgs::msg::Pose pose = geometry_msgs::msg::Pose();
							geometry_msgs::msg::Vector3 lin_vel = geometry_msgs::msg::Vector3();
							geometry_msgs::msg::Vector3 ang_vel = geometry_msgs::msg::Vector3();
							p.x = this->new_ir_beacon_pose.position.x+0.1; p.y = this->new_ir_beacon_pose.position.y; p.z = this->new_ir_beacon_pose.position.z;
							pose.position = p; pose.orientation = this->new_ir_beacon_pose.orientation;
							lin_vel = this->new_ir_beacon_twist.linear;
							ang_vel = this->new_ir_beacon_twist.angular;
							this->SetState("irlock_beacon", pose, lin_vel, ang_vel);
						}
		            } else {
		                cout << "Episode ended, you can exit from all the terminals, thank you for having played with us." << endl;
		                this->stopped = true;
		            }
                }
            };
            timer_ = this->create_wall_timer(100ms, timer_callback);
        }

    private:
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<Float32MultiArray>::SharedPtr pub_agent_vec;
        rclcpp::Subscription<Timesync>::SharedPtr timesync_sub_;
        rclcpp::Subscription<Float32MultiArray>::SharedPtr vehicle_odometry_subscriber;
        rclcpp::Publisher<Float32MultiArray>::SharedPtr play_reset_publisher;
        rclcpp::Subscription<Float32MultiArray>::SharedPtr play_reset_subscriber;
        rclcpp::Subscription<ContactsState>::SharedPtr bumper_subscriber;
        std::shared_ptr<rclcpp::Client<gazebo_msgs::srv::GetEntityState>> get_state_client_;
  		std::shared_ptr<rclcpp::Client<gazebo_msgs::srv::SetEntityState>> set_state_client_;

        void vehicle_odometry_callback(const Float32MultiArray::SharedPtr msg);
        void play_reset_callback(const Float32MultiArray::SharedPtr msg_float);
        void check_landed_callback(const ContactsState::SharedPtr msg_float);
        void GetState(const std::string & _entity);
        			  
        void SetState(const std::string & _entity, const geometry_msgs::msg::Pose & _pose, 
        			  const geometry_msgs::msg::Vector3 & _lin_vel, const geometry_msgs::msg::Vector3 & _ang_vel);

        std::atomic<uint64_t> timestamp_;
        
        float x_pos = 0.0;
        float y_pos = 0.0;
        float z_pos = 0.11;  // Drone height ~ 0.10m
        float x_drone = 0.0;
        float y_drone = 0.0;
        float z_drone = 0.0;
        int reset = 1;
        bool landed = false;
        bool stopped = false;
        bool landed_received = false;
        Float32MultiArray float32Msg = Float32MultiArray();
        std::vector<float> float32Vector = std::vector<float>(3);
        geometry_msgs::msg::Pose new_ir_beacon_pose;
        geometry_msgs::msg::Twist new_ir_beacon_twist;
        bool success_set_new_state = true;
        bool success_get_new_state = true;
};

void BaselinePrecLandNode::vehicle_odometry_callback(const Float32MultiArray::SharedPtr msg)
{
    this->z_drone = msg->data[2];
    this->x_drone = msg->data[0];
    this->y_drone = msg->data[1];
}

void BaselinePrecLandNode::play_reset_callback(const Float32MultiArray::SharedPtr msg_float)
{
    this->reset = msg_float->data[1];  // if 1 performing takeoff, 0 takeoff complete

    if(this->reset == 0) {
        this->float32Vector.clear();
        this->float32Vector.push_back(1);  // Play to 1
        this->float32Vector.push_back(0);  // Reset to 0

        this->float32Msg.data = this->float32Vector;
        this->play_reset_publisher->publish(this->float32Msg);
    }
}
        
void BaselinePrecLandNode::check_landed_callback(const ContactsState::SharedPtr msg)
{
    this->landed = msg->states.size()>0;
    this->landed_received = true;
}

/// Helper function to call get state service
void BaselinePrecLandNode::GetState(const std::string & _entity) {
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


/// Helper function to call set state service
void BaselinePrecLandNode::SetState(const std::string & _entity, 
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
    cout << "Starting baseline precision landing node" << endl;
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
	rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BaselinePrecLandNode>());
    rclcpp::shutdown();

    return 0;
}

