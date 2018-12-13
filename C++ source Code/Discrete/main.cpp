#include <iostream>
#include "DTree2.hpp"
using std::cout; using std::cin;
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
	DTree myDecisionTree(4, 5, 0.01);
	myDecisionTree.buildTree();
}
