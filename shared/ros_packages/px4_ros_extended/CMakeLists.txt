cmake_minimum_required(VERSION 3.5)
project(px4_ros_extended)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(rclpy REQUIRED)

get_filename_component(PX4_MSGS_DIR "../" ABSOLUTE BASE_DIR ${px4_msgs_DIR})
set(MSGS_DIR "${PX4_MSGS_DIR}/msg" CACHE INTERNAL "MSGS_DIR")

# environment
add_executable(env src_cpp/env.cpp)
ament_target_dependencies(env rclcpp px4_msgs std_msgs gazebo_msgs)

# baseline prec landing
add_executable(baseline_prec_land src_cpp/baseline_prec_land.cpp)
ament_target_dependencies(baseline_prec_land std_msgs rclcpp px4_msgs gazebo_msgs)

ament_python_install_package(src_py)

install(TARGETS
    env
    baseline_prec_land
    DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
    src_py/env_wrapper.py
    src_py/ddpg_agent.py
    src_py/test_ddpg_agent.py
    src_py/gazebo_runner.py
    src_py/test_env_wrapper.py
    DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
