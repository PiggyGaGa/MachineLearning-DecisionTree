#pragma once
#include "DTreeClass.h"

namespace machinelearning
{
	DecisionTree::DecisionTree()
	{
		this->root = new TreeNode;
		this->trainMatrixInfo.cols = 0;
		this->trainMatrixInfo.rows = 0;

	}
	DecisionTree::TreeNode* DecisionTree::BuildTree(cv::Mat &data, vector<Attr> &dataAttr, string Algorithm)
	{
		if (Algorithm == "ID3")
		{
			return AlgorithmID3(data, dataAttr);
		}
		else if (Algorithm == "C4.5")
		{
			return AlgorithmC4_5(data, dataAttr);
		}
		else if (Algorithm == "CART")
		{
			return AlgorithmCART(data, dataAttr);
		}
		else
		{
			cerr << "Input error, the program has been stopped!";
			exit(0);
		}
		return 0;
	}
	int DecisionTree::ReadTrainDataFile(string fileAddress)
	{
		int cols = 0, rows = 0;  //通过文件读取 获取行和列的信息。
								 ///cv::Mat strResult;
								 //cv::Mat result;    //返回的特征值矩阵。
		vector<vector<string>> strResult;
		vector<Attr> headAttr;
		ifstream read;
		read.open(fileAddress, ios::in);

		string headline;
		getline(read, headline);
		rows++;  //每次readline都要加行数一次
		if (rows > 0)
		{
			string attributeName;
			char delim = ',';
			istringstream stringin(headline);
			vector<string> oneLineString;
			while (getline(stringin, attributeName, delim))
			{
				cols++;
				oneLineString.push_back(attributeName);
			}
			strResult.push_back(oneLineString);
			//cout << endl;
		}
		//获取第一行信息
		string line;   //从第二行以后开始存储的每一行的数据
					   //下面获取后面的信息		
		while (getline(read, line))
		{
			rows++;
			int realcols = 0;  //用来判断此行中数据是否和标题行的数据列数相同，相同才存储，否则不存储
			string attributeName;
			char delim = ',';
			istringstream stringin(line);
			vector<string> oneLineString;
			while (getline(stringin, attributeName, delim))
			{
				realcols++;
				//cout << attributeName << "  ";
				oneLineString.push_back(attributeName);
			}
			if (realcols == cols)
			{ //只有和标题行数据列数相同才存储
				strResult.push_back(oneLineString);
			}
		}
		//cout << rows << "  " << cols;
		cv::Mat tmpMat(rows - 1, cols, CV_8UC1);
		this->trainMatrixInfo.cols = cols;
		this->trainMatrixInfo.rows = rows - 1;
		stringDataToInt(strResult, tmpMat, headAttr);
		this->trainDataMat = tmpMat;
		this->vectorAttr = headAttr;
		return 0;
	}
	int DecisionTree::stringDataToInt(vector<vector<string>> src, cv::Mat &dataMat, vector<Attr> &headAttrInfo)
	{
		int matRows = int(src.size());
		int matCols = int(src[0].size());
		for (int j = 0; j < matCols; j++)
		{
			vector<string> perColStringVector; //每一列的z字符串种类
			map<string, unsigned char> oneColMap;
			unsigned char indexNum = 0;  //这个数用来产生整型的参数值。
			Attr perColAttr;
			perColAttr.Attribute = src[0][j];
			perColAttr.colIndex = j;
			for (int i = 1; i < matRows; i++)
			{
				if (StringExistInVector(src[i][j], perColStringVector))
				{
					dataMat.at<unsigned char>(i - 1, j) = oneColMap[src[i][j]];
				}
				else
				{
					dataMat.at<unsigned char>(i - 1, j) = indexNum;
					perColStringVector.push_back(src[i][j]);
					oneColMap.insert(map<string, unsigned char>::value_type(src[i][j], indexNum));
					indexNum++;
				}
			}
			perColAttr.typeMap = oneColMap;
			perColAttr.typeNum = indexNum;
			perColAttr.AttributeValue = perColStringVector;
			headAttrInfo.push_back(perColAttr);
		}
		return 0;
	}

	bool DecisionTree::StringExistInVector(string str, vector<string> strVector)
	{
		int vectorSize = int(strVector.size());
		for (int i = 0; i < vectorSize; i++)
		{
			if (strVector[i] == str)
			{
				return true;
			}
		}
		return false;
	}

	DecisionTree::TreeNode* DecisionTree::AlgorithmID3(cv::Mat &data, vector<Attr> &properties)
	{
		//对矩阵进行计算，包括entropy,informationGain;
		//vector<float> entropy; //信息熵
		vector<float> informationGain;  //信息增益
		int maxIndex;  //最大信息增益的属性的索引
					   //entropy = CalculateEntropy(data);
		//cout << "dataSize:" << data.size() << endl;
		//cout << data << endl;
		informationGain = CalculateInfGain(data);
		maxIndex = FindMaxInformationGain(informationGain);
		if (data.cols == 2)
		{
			TreeNode* leaf = new TreeNode;
			string  label;
			label = MostInMatLabel(data, properties);
			leaf->Attribute = properties[0].Attribute;
			leaf->LeafNode = true;
			return leaf;
		}
		else if (TheSameLabel(data))
		{
			TreeNode* leaf = new TreeNode;
			int labelIndex = int(properties.size() - 1);
			int labelValue = int(data.at<unsigned char>(0, data.cols - 1));
			string label = FindAttrString(labelValue, properties[labelIndex]);
			leaf->Attribute = label;
			leaf->LeafNode = true;
			return leaf;
		}
		else
		{
			//进行递归
			TreeNode* branchNode = new TreeNode;
			branchNode->Attribute = properties[maxIndex].Attribute;
			branchNode->LeafNode = false;
			vector<Attr> tmpAttr = properties;   //构建新的属性向量，用于下次
			tmpAttr.erase(tmpAttr.begin() + maxIndex);  //删除上一个节点的属性。
			vector<string> attributeValue = properties[maxIndex].AttributeValue;
			//进行分支

			for (int i = 0; i < properties[maxIndex].AttributeValue.size(); i++)
			{
				TreeNode* childNode = new TreeNode; //声明一个孩子节点
				string oneAttributeValue = attributeValue[i];  //每一个属性对应的值，字符串。
				if (DataExistAttribute(data, properties, maxIndex, oneAttributeValue))
				{
					cv::Mat subMat;
					subMat = GetNewMat(data, properties, maxIndex, oneAttributeValue);
					//这里预留位置，进行预剪枝操作，
					//预剪枝操作

					childNode = this->AlgorithmID3(subMat, tmpAttr);
					branchNode->children.push_back(childNode);
					branchNode->AttributeLinkChildren.insert(map<string, TreeNode*>::value_type(oneAttributeValue, childNode));
				}
			}
			//返回分支节点
			return branchNode;
		}
	}
	DecisionTree::TreeNode* DecisionTree::AlgorithmC4_5(cv::Mat &data, vector<Attr> &a)
	{
		return nullptr;
	}
	DecisionTree::TreeNode* DecisionTree::AlgorithmCART(cv::Mat &data, vector<Attr> &a)
	{
		return nullptr;
	}
	vector<float> DecisionTree::CalculateInfGain(cv::Mat &data) //计算InformationGain
	{

		vector<float> result;
		float dataEntropy; //矩阵的熵
		dataEntropy = GetDataEntropy(data);


		//vector<int> AttrValueNum;
		for (int i = 0; i < data.cols - 1; i++)  //只对属性列进行计算，对标签列不计算，标签列已经计算过了。
		{
			vector<int> attrValue;  //每一列数据公有几种类型存储在这里
			map<int, entropyInfo> attrLinkToLabel;  //每种类型对应的信息熵信息，有可能分类相同，有可能分类不同

			for (int j = 0; j < data.rows; j++)
			{
				int value = int(data.at<unsigned char>(j, i));
				if (IntExistInVector(value, attrValue) >= 0)
				{
					int index = IntExistInVector(int(data.at<unsigned char>(j, data.cols - 1)), attrLinkToLabel[value].labelValue);
					if (index == -1)
					{
						attrLinkToLabel[value].labelValue.push_back(int(data.at<unsigned char>(j, data.cols - 1)));
						attrLinkToLabel[value].labelValueNum.push_back(1);
					}
					else
					{
						attrLinkToLabel[value].labelValueNum[index]++;
					}
				}
				else
				{
					entropyInfo oneAttrEntropyInfo;
					attrValue.push_back(value);
					oneAttrEntropyInfo.labelValue.push_back(int(data.at<unsigned char>(j, data.cols - 1)));
					oneAttrEntropyInfo.labelValueNum.push_back(1);
					attrLinkToLabel.insert(map<int, entropyInfo>::value_type(value, oneAttrEntropyInfo));
				}
			}
			//计算信息熵
			float informationGain;  //信息熵
			informationGain = InformationGain(attrValue, attrLinkToLabel, dataEntropy, data.rows);
			result.push_back(informationGain);
		}
		return result;
	}
	int DecisionTree::FindMaxInformationGain(vector<float> s)
	{
		int maxIndex = 0;
		float max = 0;
		for (int i = 0; i < s.size(); i++)
		{
			if (s[i] > max)
			{
				max = s[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	bool DecisionTree::TheSameLabel(cv::Mat &data)
	{
		unsigned char first = data.at<unsigned char>(0, data.cols - 1);
		for (int i = 1; i < data.rows; i++)
		{
			if (data.at<unsigned char>(i, data.cols - 1) != first)
			{
				return false;
			}
		}
		return true;
	}

	cv::Mat DecisionTree::GetNewMat(cv::Mat &data, vector<Attr> &properties, int maxIndex, string oneAttributeValue)
	{
		int matCols = data.cols, matRows = data.rows;
		int dimensionOFAttr = int(properties.size());
		cv::Mat result;   //结果矩阵
		unsigned char AttrValue = properties[maxIndex].typeMap[oneAttributeValue];  //字符串属性值在矩阵中对应的数值
		if (matCols != dimensionOFAttr || maxIndex >= matCols)
		{   //做一个数据维数的判断，增加健壮性
			cerr << "矩阵维数和属性向量维数不对应";
			getchar();
			exit(0);
		}
		//下面是生成小矩阵的过程。
		for (int i = 0; i < matRows; i++)
		{
			cv::Mat_<unsigned char> newRowValue;
			if (data.at<unsigned char>(i, maxIndex) == AttrValue)  //如果这一行中的元素是字符属性对应的数值执行下面操作。
			{
				for (int j = 0; j < matCols; j++)
				{
					if (j != maxIndex)
					{
						newRowValue.push_back(data.at<unsigned char>(i, j));
					}
				}
				//压入数据
				newRowValue = newRowValue.t();
				//cout << newRowValue << endl;
				if (newRowValue.cols == matCols - 1)
				{  //按行压入
					result.push_back(newRowValue);
				}
				else
				{
					cerr << "在生成小矩阵时，向量维数和矩阵维数不一致";
				}
			}
		}
		return result;
	}

	int DecisionTree::IntExistInVector(int a, vector<int> b)
	{ //如果在里面则返回对应的索引，否则返回-1
		for (int i = 0; i < b.size(); i++)
		{
			if (a == b[i])
			{
				return i;
			}
		}
		return -1;
	}

	float DecisionTree::GetDataEntropy(cv::Mat &data)
	{
		vector<float> ratio;
		vector<int> label;
		map<int, int> labelNum;
		int labelCol = data.cols - 1;
		for (int i = 0; i < data.rows; i++)
		{
			int value = int(data.at<unsigned char>(i, labelCol));
			if (IntExistInVector(value, label) >= 0)
			{
				labelNum[value]++;
			}
			else
			{
				label.push_back(value);
				labelNum.insert(map<int, int>::value_type(value, 1));
			}
		}
		for (int i = 0; i < label.size(); i++)
		{
			ratio.push_back(float(labelNum[label[i]]) / float(data.rows));

		}
		return Entropy(ratio);
	}

	float DecisionTree::InformationGain(vector<int> value, map<int, entropyInfo> b, float dataEntropy, int matRows)
	{
		float result = 0;
		for (int i = 0; i < value.size(); i++)
		{
			int D_v = 0;
			vector<float> ratio;
			for (int j = 0; j < b[value[i]].labelValueNum.size(); j++)
			{
				D_v = D_v + b[value[i]].labelValueNum[j];
			}
			for (int j = 0; j < b[value[i]].labelValueNum.size(); j++)
			{
				ratio.push_back(float(b[value[i]].labelValueNum[j]) / float(D_v));
			}
			result = result + float(float(D_v) / float(matRows)) * Entropy(ratio);

		}
		result = dataEntropy - result;
		return result;
	}

	float DecisionTree::Entropy(vector<float> ratio)
	{
		float result = 0;
		for (int i = 0; i < ratio.size(); i++)
		{
			result = result + ratio[i] * log2(ratio[i]);
		}
		return -result;
	}
	bool DecisionTree::DataExistAttribute(cv::Mat &data, vector<Attr> &properties, int maxIndex, string oneAttributeValue)
	{
		int stringValue = properties[maxIndex].typeMap[oneAttributeValue];
		for (int i = 0; i < data.rows; i++)
		{
			int dataValue = int(data.at<unsigned char>(i, maxIndex));
			if (dataValue == stringValue)
			{
				return true;
			}
		}
		return false;
	}

	vector<vector<string>> DecisionTree::ReadPredictedDataFile(string fileAddress)
	{   //读取带预测数据
		ifstream read;
		read.open(fileAddress, ios::in);
		vector<vector<string>> predictedStringData;
		string rowString;
		while (getline(read, rowString))
		{
			string oneString;
			char dim = ',';
			istringstream stringIn(rowString);   //
			vector<string> oneRowString;
			while (getline(stringIn, oneString, dim))
			{
				oneRowString.push_back(oneString);
			}
			if (oneRowString.size() == this->trainMatrixInfo.cols - 1)
			{
				//只存储和训练数据集匹配的数据
				predictedStringData.push_back(oneRowString);
			}
			else
			{
				cerr << "待预测数据和训练数据集维数不一致" << endl;
				getchar();
				getchar();
				exit(0);
			}

		}
		this->predictedDataMat = predictedStringData;
		return predictedStringData;
	}

	vector<string> DecisionTree::Predicted(TreeNode* root, vector<vector<string>> &predictedData)
	{
		vector<string> result;
		int matCols = int(predictedData[0].size());
		int matRows = int(predictedData.size());
		for (int i = 0; i < matRows; i++)
		{
			string answer;
			answer = PredictedRecursion(root, predictedData[i], this->vectorAttr);
			result.push_back(answer);
		}
		return result;
	}
	string DecisionTree::PredictedRecursion(TreeNode* nodeAddress, vector<string> &rowData, vector<Attr> &vecAttr)
	{
		//预测的递归函数
		if (nodeAddress->LeafNode)
		{
			return nodeAddress->Attribute;
		}
		else
		{
			string nodeString = nodeAddress->Attribute;
			int attrIndex = IndexOFAttribute(nodeString, vectorAttr);
			if (attrIndex >= 0)
			{
				string attrValue = rowData[attrIndex];
				TreeNode* newRoot;
				newRoot = nodeAddress->AttributeLinkChildren[attrValue];
				return PredictedRecursion(newRoot, rowData, vecAttr);

			}
		}
		return nullptr;	
	}
	
	int DecisionTree::IndexOFAttribute(string nodeString, vector<Attr> &vectorAttr)
	{
		for (int i = 0; i < vectorAttr.size(); i++)
		{
			if (nodeString == vectorAttr[i].Attribute)
			{
				return i;
			}
		}
		return -1;
	}

	string DecisionTree::FindAttrString(int a, Attr b)
	{
		for (int i = 0; i < b.AttributeValue.size(); i++)
		{
			string tmp = b.AttributeValue[i];
			if (b.typeMap[tmp] == a)
			{
				return tmp;
			}
		}
		cout << "查找出错了" << endl;
		return nullptr;
	}
	string DecisionTree::MostInMatLabel(cv::Mat &data, vector<Attr> &properties)
	{
		class vecInf   //建立只只针对此函数的内部类，在里面实现查找的功能。
		{
		public:
			static struct data
			{
				int value;
				int num;
			};
			static int InStructVector(int v, vector<data> b)
			{
				for (int i = 0; i < b.size(); i++)
				{
					if (v == b[i].value)
					{
						return i;
					}
				}
				return -1;
			}
		};
		int rows = data.rows;
		vector<vecInf::data> intVec;
		for (int i = 0; i < rows; i++)
		{
			int labelValue = int(data.at<unsigned char>(i, data.cols - 1));
			int indexOFVec = vecInf::InStructVector(labelValue, intVec);
			if (indexOFVec >= 0)
			{
				intVec[indexOFVec].num++;
			}
			else
			{
				vecInf::data tmp;
				tmp.value = labelValue;
				tmp.num = 1;
				intVec.push_back(tmp);
			}
		 }
		int maxValue = 0, maxNum = 0;
		for (int i = 0; i < intVec.size(); i++)
		{
			if (intVec[i].num >= maxNum)
			{
				maxValue = intVec[i].value;
			}
		}
		string result = FindAttrString(maxValue, properties[properties.size() - 1]);
		return result;
	}
}

