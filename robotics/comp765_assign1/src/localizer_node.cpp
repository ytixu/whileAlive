#include <random>
#include <math.h>  

#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_listener.h>

#define max(x,y) x > y ? x : y
#define min(x,y) x < y ? x : y

#define METRE_TO_PIXEL_SCALE 50
#define FORWARD_SWIM_SPEED_SCALING 0.1
#define POSITION_GRAPHIC_RADIUS 20.0
#define HEADING_GRAPHIC_LENGTH 50.0

#define CAMERA_X_OFFSET 0.384
#define CAMERA_Y_OFFSET 0.07

#define PI 3.1415926
#define TWOPI PI*2
#define YAW_DELAY_GAIN 0.6
#define YAW_PROPOSAL_SIGMA PI/15.0
#define SPEED_PROPOSAL_SIGMA 0.05
#define CAM_FOCAL_LENGTH 238.3515418007097
#define MEASUREMENT_IMG_RADIUS 2
#define MAX_X_KIDNAPPED 5
#define MAX_Y_KIDNAPPED 15
#define MAX_THETA_KIDNAPPED PI

#define num_particles 2
#define POS_PROPOSAL_SIGMA 0.1
#define OBSERVATION_SIGMA 40.0

typedef struct particle
{
    double posX;
    double posY;
    double posRot;

    double weight;
    double newWeight; 
    double cumWeight; 
    cv::Mat observation;
}
Particle;

class Localizer
{
public:
  ros::NodeHandle nh;
  image_transport::Publisher pub;
  image_transport::Subscriber gt_img_sub;
  image_transport::Subscriber robot_img_sub;
  ros::Subscriber motion_command_sub;
  ros::Publisher estimate_pub;

  cv::Mat map_image;
  cv::Mat localization_result_image;

  geometry_msgs::PoseStamped estimated_location;

  Particle particles[num_particles];

  std::default_random_engine generator;
  std::normal_distribution<double> distribution;

  Localizer( int argc, char** argv )
  {
    initializeParticles();
    distribution = std::normal_distribution<double>(0.0,POS_PROPOSAL_SIGMA);

    image_transport::ImageTransport it(nh);

    pub = it.advertise("/assign1/localization_debug_image", 1);
    estimate_pub = nh.advertise<geometry_msgs::PoseStamped>( "/assign1/localization_estimate",1);
    std::string ag_path = ros::package::getPath("aqua_gazebo");
    map_image = cv::imread((ag_path+"/materials/fishermans_small.png").c_str(), CV_LOAD_IMAGE_COLOR);

    localization_result_image = map_image.clone();

    robot_img_sub = it.subscribe("/aqua/back_down/image_raw", 1, &Localizer::robotImageCallback, this);
    motion_command_sub = nh.subscribe<geometry_msgs::PoseStamped>("/aqua/target_pose", 1, &Localizer::motionCommandCallback, this);

    ROS_INFO( "localizer node constructed and subscribed." );
  }

  // This is the work-horse of the measurement processing. Compute the geometry and
  // call to bestMatch to compute error over a small image region. 
  double evaluateMeasurementModel( cv::Mat& robot_image, const double& x, const double& y, const double& theta )
  {
    double total_match_dist = 0.0;

    float camera_x = x - CAMERA_X_OFFSET * cos(theta) - CAMERA_Y_OFFSET*sin(theta);
    float camera_y = y - CAMERA_X_OFFSET * sin(theta) - CAMERA_Y_OFFSET*cos(theta);

    for( double probe_pixel_x=100.0; probe_pixel_x <= 300; probe_pixel_x+=100.0)
      for( double probe_pixel_y=100.0; probe_pixel_y <= 300; probe_pixel_y+=100.0)
      {
        double probe_relative_X = 2.0 / CAM_FOCAL_LENGTH * ( probe_pixel_x - robot_image.size().width/2 );
        double probe_relative_Y = 2.0 / CAM_FOCAL_LENGTH * ( probe_pixel_y - robot_image.size().height/2 );

        double probe_abs_pixel_x = map_image.size().width/2 + METRE_TO_PIXEL_SCALE * ( camera_x - probe_relative_Y * cos(theta) + probe_relative_X * sin(theta) );
        double probe_abs_pixel_y = map_image.size().height/2 + METRE_TO_PIXEL_SCALE * ( camera_y + probe_relative_Y * sin(theta) + probe_relative_X * cos(theta) );

        cv::Vec3b predicted_color = robot_image.at<cv::Vec3b>(probe_pixel_y, probe_pixel_x);

        total_match_dist += bestMatch( map_image, cv::Point( probe_abs_pixel_x, probe_abs_pixel_y), predicted_color, MEASUREMENT_IMG_RADIUS );
      }

    return total_match_dist;
  }

  double bestMatch( const cv::Mat& im, const cv::Point& pt, const cv::Vec3b& target_color, int radius )
  {
    double best_match = 999999999.0;

    int row_limit = min(im.rows,pt.y+radius+1);
    int col_limit = min(im.cols,pt.x+radius+1);

    for( int row=max(0,pt.y-radius); row<row_limit; row++ )
      for( int col=max(0,pt.x-radius); col<col_limit; col++ )
      {
        cv::Vec3b observed_color = im.at<cv::Vec3b>(row,col);
        double dist = sqrt(((int)(target_color[0]) - (int)(observed_color[0])) * ((int)(target_color[0]) - (int)(observed_color[0])) +
                     ((int)(target_color[1]) - (int)(observed_color[1])) * ((int)(target_color[1]) - (int)(observed_color[1])) +
                   ((int)(target_color[2]) - (int)(observed_color[2])) * ((int)(target_color[2]) - (int)(observed_color[2])));

        if( dist < best_match )
          best_match = dist;
      }

    return best_match;
  }

  void initializeParticles()
  {
    for (int p=0; p<num_particles; p++)
    {
      particles[p].posX = distribution(generator);
      particles[p].posY = distribution(generator);
      double angle = distribution(generator);
      particles[p].posRot =  angle - TWOPI * floor( angle / TWOPI );
      particles[p].weight = 1.0 / num_particles;
    }
  }

  void robotImageCallback( const sensor_msgs::ImageConstPtr& robo_img_msg )
  {
    cv::Mat robot_image = cv_bridge::toCvCopy(robo_img_msg, sensor_msgs::image_encodings::BGR8)->image;

    double total_weight = 0.0;

    for (int p=0; p<num_particles; p++)
    {
      // double junk_r, junk_p,estimated_yaw;
      // tf::Quaternion tf_q;
      // tf::quaternionMsgToTF(estimated_location.pose.orientation, tf_q );
      // tf::Matrix3x3(tf_q).getEulerYPR( estimated_yaw, junk_p, junk_r );
      double measurement_score = evaluateMeasurementModel(robot_image,  particles[p].posX, particles[p].posY,  particles[p].posRot);
      ROS_INFO( "LOCALIZER: The current measurement score is %f at %f, %f, %f.", measurement_score, particles[p].posX, particles[p].posY,  particles[p].posRot);

      total_weight += measurement_score;
      particles[p].newWeight = measurement_score;
    }

    double accumulated_weight = 0.0;
    int best_particle;
    double best_score = -10000;

    for (int p=0; p<num_particles; p++)
    {
      double score = particles[p].weight + log(particles[p].newWeight/total_weight);
      particles[p].weight = score;
      accumulated_weight += score;
      particles[p].cumWeight = accumulated_weight;
      if (best_score < score) 
      {
        best_score = score;
        best_particle = p;
      }
    }

    estimated_location.pose.position.x = particles[best_particle].posX;
    estimated_location.pose.position.y = particles[best_particle].posY;
    estimated_location.pose.orientation = particles[best_particle].posRot;
  }

  void motionCommandCallback(const geometry_msgs::PoseStamped::ConstPtr& motion_command )
  {
    geometry_msgs::PoseStamped command = *motion_command;
    double target_roll, target_pitch, target_yaw;
    tf::Quaternion target_orientation;
    tf::quaternionMsgToTF(command.pose.orientation, target_orientation);
    tf::Matrix3x3(target_orientation).getEulerYPR( target_yaw, target_pitch, target_roll );

    // The following three lines implement the basic motion model example
    double delta_x = FORWARD_SWIM_SPEED_SCALING * command.pose.position.x * cos( -target_yaw );
    double delta_y = FORWARD_SWIM_SPEED_SCALING * command.pose.position.x * sin( -target_yaw );
    double delta_t = command.pose.orientation - estimated_location.pose.orientation;
    estimated_location.pose.position.x = estimated_location.pose.position.x + delta_x;
    estimated_location.pose.position.y = estimated_location.pose.position.y + delta_y;
    estimated_location.pose.orientation = command.pose.orientation;

    for (int p=0; p<num_particles; p++)
    {
      particles[p].posX = particles[p].posX + delta_x;
      particles[p].posY = particles[p].posY + delta_y;
      double angle = particles[p].posRot + delta_t;
      particles[p].posRot =  angle - TWOPI * floor( angle / TWOPI );
    }

    // The remainder of this function is sample drawing code to plot your answer on the map image.

    // This line resets the image to the original map so you can start drawing fresh each time.
    // Comment the one following line to plot your whole trajectory
    localization_result_image = map_image.clone();

    int estimated_robo_image_x = localization_result_image.size().width/2 + METRE_TO_PIXEL_SCALE * estimated_location.pose.position.x;
    int estimated_robo_image_y = localization_result_image.size().height/2 + METRE_TO_PIXEL_SCALE * estimated_location.pose.position.y;

    int estimated_heading_image_x = estimated_robo_image_x + HEADING_GRAPHIC_LENGTH * cos(-target_yaw);
    int estimated_heading_image_y = estimated_robo_image_y + HEADING_GRAPHIC_LENGTH * sin(-target_yaw);

    cv::circle( localization_result_image, cv::Point(estimated_robo_image_x, estimated_robo_image_y), POSITION_GRAPHIC_RADIUS, CV_RGB(250,0,0), -1);
    cv::line( localization_result_image, cv::Point(estimated_robo_image_x, estimated_robo_image_y), cv::Point(estimated_heading_image_x, estimated_heading_image_y), CV_RGB(250,0,0), 10);

    estimate_pub.publish( estimated_location );
  }

  void spin()
  {
    ros::Rate loop_rate(30);
    while (nh.ok()) {
      sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", localization_result_image).toImageMsg();
      pub.publish(msg);

      ros::spinOnce();
      loop_rate.sleep();
    }
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "localizer");
  Localizer my_loc(argc, argv);
  my_loc.spin();
}
