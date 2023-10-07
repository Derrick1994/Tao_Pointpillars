#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include"tao_Pointpillars/bbox.h"
#include"tao_Pointpillars/result.h"

#include "cuda_runtime.h"
#include "pointpillar.h"

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

Eigen::Quaterniond euler2Quaternion(const double roll, const double pitch,
                                    const double yaw) {
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    return q;
}


class CloudProcess
{
public:
    CloudProcess(ros::NodeHandle& nh, std::string topic_sub, std::string topic_pub, size_t buff_size);
    ~CloudProcess();
    void callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg_ptr);
    // ros::Timer fuel_timer;
    void Timer();
    
    
private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;
    
    ros::Publisher pub_pointcloud;
    ros::Publisher pub_txt;
    ros::Publisher pub_bbox;
    ros::Publisher pub_heading;
    std::vector<Bndbox> nms_pred;
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaStream_t stream ;
    
    //const
    std::vector<std::string> class_names{"Car","Truck"};
    float nms_iou_thresh = 0.01;
    int pre_nms_top_n = 4096;
    bool do_profile{false};
    std::string model_path;
    std::string engine_path = "src/tao_Pointpillars/model/onnx.fp32.engine";
    std::string data_type{"fp32"};
    std::string output_path;
    
    visualization_msgs::MarkerArray bboxes;
    visualization_msgs::MarkerArray txt_markers;
    visualization_msgs::MarkerArray headings;
    
    int msgs_count = 0;
};

CloudProcess::CloudProcess(ros::NodeHandle &nh, std::string topic_sub, std::string topic_pub, size_t buff_size) {

    nh_ = nh;
    sub_ = nh_.subscribe(topic_sub, buff_size, &CloudProcess::callback, this);
    
    pub_= nh_.advertise<tao_Pointpillars::result>(topic_pub, 10);

    pub_pointcloud =nh_.advertise<sensor_msgs::PointCloud2>("/pointcloud_visualization",1);
    pub_bbox = nh_.advertise<visualization_msgs::MarkerArray>("/bbox_visualization", 1);
    pub_txt =nh_.advertise<visualization_msgs::MarkerArray>("/txt_visualization", 1);
    pub_heading =nh_.advertise<visualization_msgs::MarkerArray>("/heading_visualization", 1);

    // start stop to cal time
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    // stream to keep work order
    checkCudaErrors(cudaStreamCreate(&stream));
    nms_pred.reserve(100);
    elapsedTime = 0.0f;
    stream = NULL;

}

CloudProcess::~CloudProcess(){
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));
}

void CloudProcess::Timer(){
    sleep(0.3);
    visualization_msgs::Marker bbox;
    visualization_msgs::Marker txt_marker;
    visualization_msgs::Marker head;
    bbox.action = visualization_msgs::Marker::DELETEALL;
    txt_marker.action = visualization_msgs::Marker::DELETEALL;
    head.action = visualization_msgs::Marker::DELETEALL;
    bboxes.markers.push_back(bbox);
    txt_markers.markers.push_back(txt_marker);
    headings.markers.push_back(head);
    pub_bbox.publish(bboxes);
    pub_txt.publish(txt_markers);
    pub_heading.publish(headings);
    
}

void CloudProcess::callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg_ptr) {
    msgs_count += 1;

    nms_pred.clear();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::fromROSMsg(*cloud_msg_ptr, *cloud);
    float* points_ptr = new float[cloud->points.size() * 4];
    for(int i=0;i<cloud->points.size();i++){
        points_ptr[i*4]=cloud->points[i].x;
        points_ptr[i*4+1]=cloud->points[i].y;
        points_ptr[i*4+2]=cloud->points[i].z;
        points_ptr[i*4+3]=cloud->points[i].intensity;
    }

    PointPillar pointpillar(model_path, engine_path, stream, data_type);

    //float* points = (float*)buffer.get();
    unsigned int num_point_values = pointpillar.getPointSize();
    unsigned int points_size = cloud->points.size();
    

    float *points_data = nullptr;
    unsigned int *points_num = nullptr;
    unsigned int points_data_size = points_size * num_point_values * sizeof(float);
    
    checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
    checkCudaErrors(cudaMallocManaged((void **)&points_num, sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(points_data, points_ptr, points_data_size, cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(points_num, &points_size, sizeof(unsigned int), cudaMemcpyDefault));
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventRecord(start, stream);

    pointpillar.doinfer(
      points_data, points_num, nms_pred,
      nms_iou_thresh,
      pre_nms_top_n,
      class_names,
      do_profile
    );
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "TIME: pointpillar: " << elapsedTime << " ms." << std::endl;
    
    checkCudaErrors(cudaFree(points_data));
    checkCudaErrors(cudaFree(points_num));
    
    tao_Pointpillars::result m_result;
    for(auto box :nms_pred){
        tao_Pointpillars::bbox m_box;
        if(box.score>0.3){
            m_box.x = box.x;
            m_box.y = box.y;
            m_box.z = box.z;
            m_box.l = box.l;
            m_box.w = box.w;
            m_box.h = box.h;
            m_box.rt = box.rt;
            m_box.id = box.id;
            m_box.score = box.score;
            
            m_result.result.push_back(m_box);
        }
    }

    
    // bbox.action = visualization_msgs::Marker::DELETEALL;
    
    bboxes.markers.clear();
    for (int j = 0; j < m_result.result.size(); j++) {
        visualization_msgs::Marker bbox;
        // get cube
        bbox.header.frame_id = "CLR";
        bbox.header.stamp = ros::Time::now();
        bbox.ns = "lidar";
        bbox.id = j;
        bbox.type = visualization_msgs::Marker::CUBE;
        bbox.action = visualization_msgs::Marker::ADD;
        bbox.scale.x = m_result.result[j].l;
        bbox.scale.y = m_result.result[j].w;
        bbox.scale.z = m_result.result[j].h;
        bbox.color.a = 0.5;
        bbox.color.r = 0;
        bbox.color.g = 1;
        bbox.color.b = 0;
        bbox.pose.position.x = m_result.result[j].x;
        bbox.pose.position.y = m_result.result[j].y;
        bbox.pose.position.z = m_result.result[j].z;
        Eigen::Quaterniond orin = euler2Quaternion(0, 0, m_result.result[j].rt);
        bbox.pose.orientation.x = orin.x();
        bbox.pose.orientation.y = orin.y();
        bbox.pose.orientation.z = orin.z();
        bbox.pose.orientation.w = orin.w();
        bboxes.markers.push_back(bbox);
    }

    // txt marker
    
    // txt_marker.action = visualization_msgs::Marker::DELETEALL;
    
    txt_markers.markers.clear();
    for (int h = 0; h < m_result.result.size(); h++) {
        visualization_msgs::Marker txt_marker;
        // get txt marker
        txt_marker.header.frame_id = "CLR";
        txt_marker.ns = "lidar";
        txt_marker.id = h;
        txt_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        txt_marker.action = visualization_msgs::Marker::ADD;
        txt_marker.scale.z = 0.4;
        txt_marker.color.a = 1.0;
        txt_marker.color.r = 1.0;
        txt_marker.color.g = 1.0;
        txt_marker.color.b = 1.0;
        txt_marker.pose.position.x = m_result.result[h].x;
        txt_marker.pose.position.y = m_result.result[h].y;
        txt_marker.pose.position.z = m_result.result[h].z + m_result.result[h].h + 1;
        txt_marker.pose.orientation.w = 1.0;
        std::string str0 = std::to_string(m_result.result[h].x);
        std::string str1 = std::to_string(m_result.result[h].y);
        std::string str2 = std::to_string(m_result.result[h].z);
        std::string str3 = std::to_string(m_result.result[h].w);
        std::string str4 = std::to_string(m_result.result[h].l);
        std::string str5 = std::to_string(m_result.result[h].h);
        std::string str6 = std::to_string(m_result.result[h].rt);
        std::string str7 = std::to_string(m_result.result[h].id);
        std::string str8 = std::to_string(m_result.result[h].score);
        std::string str_text = "x : " + str0 + "\n" + "y : " + str1 + "\n" +
                          "z : " + str2 + "\n" + "width : " + str3 + "\n" +
                          "length : " + str4 + "\n" + "height : " + str5 +
                          "\n" + "radian : " + str6 + "\n" + "type : " + str7 +
                          "\n" + "score : " + str8 + "\n";
        txt_marker.text = str_text;
        txt_markers.markers.push_back(txt_marker);
    }

    // arrow marker
    
    // head.action = visualization_msgs::Marker::DELETEALL;
    
    headings.markers.clear();
    for (int k = 0; k < m_result.result.size(); k++) {
        visualization_msgs::Marker head;
        // get arrow
        head.header.frame_id = "CLR";
        head.ns = "lidar";
        head.id = k;
        head.type = visualization_msgs::Marker::ARROW;
        head.action = visualization_msgs::Marker::ADD;
        head.scale.x = 0.2;
        head.scale.y = 0.4;
        head.scale.z = 0.4;
        head.color.a = 0.5;
        head.color.g = 1;

        geometry_msgs::Point p1, p2;
        p1.x = m_result.result[k].x;
        p1.y = m_result.result[k].y;
        p1.z = m_result.result[k].z;
        p2.x = m_result.result[k].x +
               cos(m_result.result[k].rt) * (m_result.result[k].l / 2);
        p2.y = m_result.result[k].y +
               sin(m_result.result[k].rt) * (m_result.result[k].l / 2);
        p2.z = m_result.result[k].z;
        head.points.push_back(p1);
        head.points.push_back(p2);
        // head.lifetime = ros::Duration(0.3);
        headings.markers.push_back(head);
        head.points.clear();
    }
    


    pub_pointcloud.publish(cloud_msg_ptr);
    pub_bbox.publish(bboxes);
    pub_txt.publish(txt_markers);
    pub_heading.publish(headings);
    pub_.publish(m_result);
    Timer();  //deleteall
    


    free(points_ptr);
    
    nms_pred.clear();
    std::cout << "msgs_count:::"<< msgs_count << std::endl;
}

int main(int argc, char **argv){
    
    
    ros::init(argc, argv, "cloudProcess_node");
    
    ros::NodeHandle nh("~");
    
    CloudProcess cloud_process(nh, "/zone3/lidar/pointcloud2", "/cloudProcess_pub", 1);
    
    ros::spin();
    return 0;
}
