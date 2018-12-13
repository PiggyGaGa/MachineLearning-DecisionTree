#ifndef DTREE_H
#define DTREE_H

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


class DTree
{
public:
	struct TreeNode
	{   //节点信息
		string Attribute;   //此节点对应的属性
		bool LeafNode; //如果是叶子节点，此值反映分类结果。 //其他情况都是0;
		vector<TreeNode*> children; //孩子节点的地址。
		map<string, TreeNode*> AttributeLinkChildren;  //属性指向孩子节点，也就是对应的树形结构中的树枝

	};
	struct Attr    //每一列的属性
	{
		int colIndex;
		string Attribute;
		int typeNum;   //属性取值的个数
		vector<string> AttributeValue;
		map<string, unsigned char>  typeMap; //属性取值对应的整数值;
	};
    
private:
	struct MatInfo
	{
		int cols;
		int rows;

	};		
	struct entropyInfo
	{
		vector<int> labelValue;
		vector<int> labelValueNum;
	};
public:
	//MatrixXi trainDataMat;
    MatrixXi trainDataMat;
	vector<vector<string>> predictedDataMat;
	MatInfo trainMatrixInfo;
	TreeNode *root;  //根节点
	vector<Attr> vectorAttr; //存储所有的矩阵信息，但不存储矩阵。
	DTree();
    DTree(string csvfilename);
	MatrixXi ReadTrainDataFile(string fileAddress);  //数据预处理
	//TreeNode* BuildTree(MatrixXi &data, vector<Attr> &dataAttr, string AlgorithmName);  // 指定是哪种算法
    int BuildTree(MatrixXi &data, vector<Attr> &dataAttr, string AlgorithmName);
	vector<vector<string>> ReadPredictedDataFile(string fileAddreess);
	//vector<string> Predicted(TreeNode* root, vector<vector<string>> &pData);  //返回值为int类型表示数据的分类。
    vector<string> Predicted(vector<vector<string>> &pData);
    ~DTree();
	
private:
    int stringDataToInt(const vector<vector<string>> &src, MatrixXi &dataMat, vector<Attr> &headAttrInfo);
	bool StringExistInVector(string aa, vector<string> A);
	TreeNode* AlgorithmID3(MatrixXi &data, vector<Attr> &a);
	TreeNode* AlgorithmC4_5(MatrixXi &data, vector<Attr> &a);
	TreeNode* AlgorithmCART(MatrixXi &data, vector<Attr> &a);
	//vector<float> CalculateEntropy(MatrixXi a);
	vector<float> CalculateInfGain(MatrixXi &a);
	int FindMaxInformationGain(vector<float> s);
	bool TheSameLabel(MatrixXi &a);
	MatrixXi GetNewMat(MatrixXi &a, vector<Attr> &properties, int maxIndex, string oneAttributeValue);
	int IntExistInVector(int a, vector<int> b);
	float GetDataEntropy(MatrixXi &data);
	float InformationGain(vector<int> value, map<int, entropyInfo> b, float dataEntropy, int rows);
	float Entropy(vector<float> ratio); // 计算熵
	//判断矩阵中是否有该属性
	bool DataExistAttribute(MatrixXi &data, vector<Attr> &properties, int maxIndex, string oneAttributeValue);
	int IndexOFAttribute(string nodeString, vector<Attr> &vectorAttr);
	string PredictedRecursion(TreeNode* nodeAddress, vector<string> &rowData, vector<Attr> &vecAttr);
	string FindAttrString(int a, Attr b);
	string MostInMatLabel(MatrixXi &data, vector<Attr> &properties);
    int destroyTree(TreeNode *root);
};

#endif