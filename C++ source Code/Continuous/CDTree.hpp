#ifndef CDTree_H
#define CDTree_H

#include <iostream>
#include <fstream>  //读取文件内容
#include <map>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <cmath>

using std::cout; using std::cin; using std::cerr; using std::endl;
using std::string; using std::ifstream; using std::istringstream;
using std::vector; using std::map; using std::ios;

using Eigen::MatrixXf;   //Eigen Matrix float 存储数据  训练数据
using Eigen::MatrixXi;  //Eigen Matrix int 存储数据  训练数据和标签
using Eigen::VectorXi;
using Eigen::VectorXf;


class CDTree
{		
public:
    struct TreeNode
    {   //节点信息
        TreeNode* parents;  //当前节点的双亲节点
        int AttributeIndex;   //此节点对应的属性索引
        bool LeafNode; //判断是否为叶子节点
        vector<TreeNode*> children; //孩子节点的地址。
        int label;  //if it is a leaf node then this prameter is useful

    };

    struct IntervalTree
    {
        IntervalTree *leftTree;
        IntervalTree *rightTree;
        bool LeafNode;
        float cutValue;
        IntervalTree *parients;
    };

    vector<int> ColsIndex;  //所有数据的列
    TreeNode* root;
    int deepestTree; // deepest depth of the tree
    int max_bin;   // max number of interval on every  continous parameter's cut
    float thresholdInfoGain; //在计算
    int trainX_Dimension;


	CDTree(int deepestTree, int max_bin, float thresholdInfoGain);
    CDTree();
    int buildTree(const MatrixXf &trainX, const MatrixXi &trainY, const MatrixXf &validateX, const MatrixXi &validateY, string Algorithm);
    int buildTree(const MatrixXf &trainX, const MatrixXi &trainY, string Algorithm);
    MatrixXi predict(const MatrixXf &testX);
    ~CDTree();

private:
    vector<vector<std::pair<float, float>>> intervals;  // 每个属性的区间分割
    vector<vector<std::pair<float, float>>> continous2discrete(const MatrixXf &trainX, const MatrixXi &trainY);
    IntervalTree* MultiwayPartitionGain(IntervalTree *parientsNode, const vector<float> &continuousVec, const MatrixXf &trainX, const MatrixXi &trainY, int index, float thresholdInfoGain);
    int getIntervalsFromTreeStruct(IntervalTree *Intervalroot, vector<float> &intervals);
    int sortVectorXf(const VectorXf &vec, VectorXf &sorted_vec, VectorXi &ind);
    int cutBranches(const MatrixXf &validateX, const MatrixXi &validateY);
    int cutBranches(int deepthTree);
    TreeNode* AlgorithmID3(TreeNode *parients, const MatrixXf &trainX, const MatrixXi &trainY, vector<int> attrrbuteIndexOfCurrentData);
	TreeNode* AlgorithmC4_5(const MatrixXf &trainX, const MatrixXi &trainY);
	TreeNode* AlgorithmCART(const MatrixXf &trainX, const MatrixXi &trainY);
    std::pair<MatrixXf, MatrixXi> splitData(const MatrixXf &trainX, const MatrixXi &trainY, std::pair<float, float> smallInterval, int maxIndex, string flag); 
    int FindMaxInformationGain(vector<float> s);
    vector<float> CalculateInfGain(const MatrixXf &trainX, const MatrixXi &trainY, vector<int> attributeIndexOfCurrentData);
    vector<float> getSampleProbability(const MatrixXi &trainY);
    vector<float> removeColRepetitionValue(const MatrixXf &Col);
    MatrixXi findDvlabel(const MatrixXf &trainX, const MatrixXi &trainY, int index, std::pair<float, float> smallInterval, string flag);
    float calculateEntropy(vector<float> probability);
    bool TheSameLabel(const MatrixXi &trainY);
    int chooseMostLabel(const MatrixXi &trainY);
    int findExistandDealwith(int label, vector<std::pair<int, int>> &labelAndNum);
    int destroyTree(TreeNode* root);
    int destroyIntervalTree(IntervalTree *root);

    template<typename Type>
    int KMeans(vector<Type> vec, vector<Type> &newvec, int index);
    int predictTree(TreeNode *node, const MatrixXf testX);

};

#endif