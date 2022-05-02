#include <iostream>
#include <stdint.h>
#include <chrono>
#include <math.h>

#include <rclcpp/rclcpp.hpp>

#include "std_msgs/msg/float32_multi_array.hpp"
#include <px4_msgs/msg/timesync.hpp>

#include "gazebo_msgs/msg/contacts_state.hpp"
#include <gazebo_msgs/srv/get_entity_state.hpp>
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"


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

            timesync_sub_ = this->create_subscription<Timesync>(
                "fmu/timesync/out", 2,
                [this](const Timesync::UniquePtr msg) 
                {
                    timestamp_.store(msg->timestamp);
                });            
                
            auto model_timer_callback = [this]() -> void {
                this->GetState("irlock_beacon");
                this->GetState("iris_irlock");
            };

            // Main loop
            auto timer_callback = [this]() -> void {
                if(this->reset == 0 and !this->stopped) {  // Playing, if 1 takeoff
		            landed_received = false;
		            
		            if (!this->landed) {
				        this->float32Vector.clear();
				        this->float32Vector.push_back(2.0 * (this->x_drone_world - this->ir_beacon_pose.position.x));
				        this->float32Vector.push_back(2.0 * (this->y_drone_world - this->ir_beacon_pose.position.y));
				        this->float32Vector.push_back(-0.1);
				        
				        cout<<this->x_drone_world<<" "<<this->ir_beacon_pose.position.x<<endl;
				        cout<<this->y_drone_world<<" "<<this->ir_beacon_pose.position.y<<endl;

				        this->float32Msg.data = this->float32Vector;
				        cout << "Pos (x="<<(this->x_drone_world - this->ir_beacon_pose.position.x)<<", y="<<(this->y_drone_world - this->ir_beacon_pose.position.y)<<", z="<<this->z_drone_world<<")"<< endl;
				        cout << "Vel (x="<<2.0 * (this->x_drone_world - this->ir_beacon_pose.position.x)<<", y="<<2.0 * (this->y_drone_world - this->ir_beacon_pose.position.y)<<", z="<<-0.1<<")"<< endl;
				        this->pub_agent_vec->publish(this->float32Msg);
		            } else {
		                cout << "Episode ended, you can exit from all the terminals, thank you for having played with us." << endl;
		                this->stopped = true;
		            }
                }
            };
            timer_target_ = this->create_wall_timer(50ms, model_timer_callback);  // 20 Hz
            timer_ = this->create_wall_timer(100ms, timer_callback);  // 10Hz
        }

    private:
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::TimerBase::SharedPtr timer_target_;
        rclcpp::Publisher<Float32MultiArray>::SharedPtr pub_agent_vec;
        rclcpp::Subscription<Timesync>::SharedPtr timesync_sub_;
        rclcpp::Subscription<Float32MultiArray>::SharedPtr vehicle_odometry_subscriber;
        rclcpp::Publisher<Float32MultiArray>::SharedPtr play_reset_publisher;
        rclcpp::Subscription<Float32MultiArray>::SharedPtr play_reset_subscriber;
        rclcpp::Subscription<ContactsState>::SharedPtr bumper_subscriber;
        std::shared_ptr<rclcpp::Client<gazebo_msgs::srv::GetEntityState>> get_state_client_;

        void vehicle_odometry_callback(const Float32MultiArray::SharedPtr msg);
        void play_reset_callback(const Float32MultiArray::SharedPtr msg_float);
        void check_landed_callback(const ContactsState::SharedPtr msg_float);
        void GetState(const std::string & _entity);

        std::atomic<uint64_t> timestamp_;
        
        float x_drone = 0.0;
        float y_drone = 0.0;
        float z_drone = 0.0;
        float x_drone_world = 0.0;
        float y_drone_world = 0.0;
        float z_drone_world = 0.0;
        int reset = 1;
        bool landed = false;
        bool stopped = false;
        bool landed_received = false;
        Float32MultiArray float32Msg = Float32MultiArray();
        std::vector<float> float32Vector = std::vector<float>(3);
        geometry_msgs::msg::Pose ir_beacon_pose;
        geometry_msgs::msg::Twist ir_beacon_twist;
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

// Helper function to call get state service
void BaselinePrecLandNode::GetState(const std::string & _entity) {
  auto request = std::make_shared<gazebo_msgs::srv::GetEntityState::Request>();
  request->name = _entity;

  auto target_response_received_callback = [this](rclcpp::Client<gazebo_msgs::srv::GetEntityState>::SharedFuture future) {
  	this->ir_beacon_pose.position.x = future.get()->state.pose.position.y;
    this->ir_beacon_pose.position.y = future.get()->state.pose.position.x;
    this->ir_beacon_pose.position.z = -1.0*future.get()->state.pose.position.z;
    this->ir_beacon_pose.orientation = future.get()->state.pose.orientation;
    this->ir_beacon_twist.linear.x = future.get()->state.twist.linear.y;
    this->ir_beacon_twist.linear.y = future.get()->state.twist.linear.x;
    this->ir_beacon_twist.linear.z = -1.0*future.get()->state.twist.linear.z;
    this->ir_beacon_twist.angular = future.get()->state.twist.angular;
  };
  
  auto drone_response_received_callback = [this](rclcpp::Client<gazebo_msgs::srv::GetEntityState>::SharedFuture future) {
  	this->z_drone_world = -1.0*future.get()->state.pose.position.z;
    this->x_drone_world = future.get()->state.pose.position.y;
    this->y_drone_world = future.get()->state.pose.position.x;
  };
  
  if (_entity=="iris_irlock")
  	auto response_future = get_state_client_->async_send_request(request, drone_response_received_callback);
  else if (_entity=="irlock_beacon")
  	auto response_future = get_state_client_->async_send_request(request, target_response_received_callback);
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

