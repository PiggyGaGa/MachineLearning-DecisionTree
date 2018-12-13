#include <iostream>
#include "CDTree.hpp"
#include "RFCSV.hpp"
#include <Eigen/Dense>
using Eigen::MatrixXf; using Eigen::MatrixXi;
int main()
{
/*
	string fileAddress("./tianqi.csv");
	//cout << "请输入文件的地址：";
	//cin >> fileAddress;
	//cout << sizeof(unsigned int) << endl;
	DTree myDecisionTree;
	myDecisionTree.ReadTrainDataFile(fileAddress);
	myDecisionTree.BuildTree(myDecisionTree.trainDataMat, myDecisionTree.vectorAttr, "ID3");

vector<string> result;
	vector<vector<string>> predictedData;
	predictedData = myDecisionTree.ReadPredictedDataFile("pre.csv");
	result = myDecisionTree.Predicted(predictedData);
	cout << "预测的结果如下：" << endl;
	for (size_t i = 0; i < result.size(); i++)
	{
		cout << result[i] << endl;
	} 
	return 0;
*/
	std::pair<MatrixXf, MatrixXi> trainData, testData;
	RFCSV<MatrixXf, MatrixXi> readcsv;
	trainData = readcsv.getData("../data/ContinuousTrain.csv");
	//cout << trainData.first << endl;
	//cout << trainData.second << endl;
	testData = readcsv.getData("../data/ContinuousTest.csv");
	CDTree myCDTree(4, 3, 0.0001);
	myCDTree.buildTree(trainData.first, trainData.second, "ID3");
	MatrixXi predictresult = myCDTree.predict(testData.first);
	float accuracy = 0;
	for (int i = 0; i < predictresult.size(); i++)
	{
		if (testData.second(i) == predictresult(i))
		{
			accuracy += 1.0 / predictresult.size();
		}
	}
	cout << "准确率为：" << accuracy * 100 << "\% 你很棒棒哦！" << endl;
/*
	cout << "原始标签" << endl;
	cout << testData.second << endl;
	cout << "预测标签" << endl;
	cout << predictresult << endl;
*/	
}
