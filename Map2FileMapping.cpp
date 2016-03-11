#include "Map2FileMapping.h"


Map2FileMapping::SubFieldRect::SubFieldRect(int rangeStart, int rangeEnd, int rowStart, int rowEnd)
{
	setParam(rangeStart, rangeEnd, rowStart, rowEnd);
}

void Map2FileMapping::SubFieldRect::setParam(int rangeStart, int rangeEnd, int rowStart, int rowEnd)
{
	_rangeStart = rangeStart;
	_rangeEnd = rangeEnd;
	_rowStart = rowStart;
	_rowEnd = rowEnd;

	_ranges = rangeEnd - rangeStart + 1;
	_rows = rowEnd - rowStart + 1;
	_sum = _ranges*_rows;
}

Map2FileMapping::FieldConfig::FieldConfig()
{
	// type for rep 1
	typeVec.push_back(SFRVec_short);
	typeVec.push_back(SFRVec_tall);
	typeVec.push_back(SFRVec_ps);
	repVec.push_back(typeVec);
	repVec.push_back(typeVec);
}

bool Map2FileMapping::FieldConfig::addTypeSubFieldRect(int rep, int type, SubFieldRect sfr)
{
	if (rep < 0 || rep > repVec.size() - 1)
	{
		std::cout << "No such rep stored!" << std::endl;
		return false;
	}

	switch (type)
	{
	case SHORT_PLANT:
		repVec[rep][SHORT_PLANT].push_back(sfr);
		break;
	case TALL_PLANT:
		repVec[rep][TALL_PLANT].push_back(sfr);
		break;
	case PS_PLANT:
		repVec[rep][PS_PLANT].push_back(sfr);
		break;
	default:
		break;
	}

	return true;
}

Map2FileMapping::Map2FileMapping()
{
	// Agronomy hedge
	AHConfig.name = AH;
	AHConfig.rangeStart = 2;
	AHConfig.rangeEnd = 54;
	AHConfig.rowStart = 3;
	AHConfig.rowEnd = 52;
	AHConfig.id = 1;

	// rep 1 SHORT_PLANT 
	SubFieldRect SFR(2, 8, 3, 52);
	AHConfig.addTypeSubFieldRect(REP_1, SHORT_PLANT, SFR);
	//rep 1 TALL_PLANT
	SFR.setParam(9, 15, 3, 52);
	AHConfig.addTypeSubFieldRect(REP_1, TALL_PLANT, SFR);
	//rep 1 PS_PLANT 
	SFR.setParam(16, 27, 3, 52);
	AHConfig.addTypeSubFieldRect(REP_1, PS_PLANT, SFR);
	// rep2 SHORT_PLANT 
	SFR.setParam(48, 54, 3, 52);
	AHConfig.addTypeSubFieldRect(REP_2, SHORT_PLANT, SFR);
	// rep2 TALL_PLANT 
	SFR.setParam(29, 35, 3, 52);
	AHConfig.addTypeSubFieldRect(REP_2, TALL_PLANT, SFR);
	// rep 2 PS_PLANT 
	SFR.setParam(36, 47, 3, 52);
	AHConfig.addTypeSubFieldRect(REP_2, PS_PLANT, SFR);

	fieldConfigVec.push_back(AHConfig);

	// Curtiss hedge part 1
	CHConfigPt1.name = CH1;
	CHConfigPt1.rangeStart = 2;
	CHConfigPt1.rangeEnd = 39;
	CHConfigPt1.rowStart = 3;
	CHConfigPt1.rowEnd = 48;
	CHConfigPt1.id = 3;

	// rep 1 SHORT_PLANT 
	SFR.setParam(2, 9, 3, 48);
	CHConfigPt1.addTypeSubFieldRect(REP_1, SHORT_PLANT, SFR);
	//rep 1 TALL_PLANT 
	SFR.setParam(9, 16, 3, 48);
	CHConfigPt1.addTypeSubFieldRect(REP_1, TALL_PLANT, SFR);
	//rep 1 PS_PLANT 
	SFR.setParam(16, 28, 3, 48);
	CHConfigPt1.addTypeSubFieldRect(REP_1, PS_PLANT, SFR);
	// rep2 SHORT_PLANT 
	SFR.setParam(28, 35, 3, 48);
	CHConfigPt1.addTypeSubFieldRect(REP_2, SHORT_PLANT, SFR);
	// rep2 TALL_PLANT 
	SFR.setParam(35, 39, 3, 48);
	CHConfigPt1.addTypeSubFieldRect(REP_2, TALL_PLANT, SFR);

	fieldConfigVec.push_back(CHConfigPt1);

	// Curtiss hedge part 2
	CHConfigPt2.name = CH2;
	CHConfigPt2.rangeStart = 2;
	CHConfigPt2.rangeEnd = 18;
	CHConfigPt2.rowStart = 53;
	CHConfigPt2.rowEnd = 92;
	CHConfigPt2.id = 3;

	// rep2 TALL_PLANT 
	SFR.setParam(2, 4, 53, 92);
	CHConfigPt2.addTypeSubFieldRect(REP_2, TALL_PLANT, SFR);
	// rep2 PS_PLANT 
	SFR.setParam(4, 18, 53, 92);
	CHConfigPt2.addTypeSubFieldRect(REP_2, PS_PLANT, SFR);

	fieldConfigVec.push_back(CHConfigPt2);

	// Agronomy single 
	ASConfig.name = AS;
	ASConfig.rangeStart = 1;
	ASConfig.rangeEnd = 69;
	ASConfig.rowStart = 1;
	ASConfig.rowEnd = 18;
	ASConfig.id = 2;

	fieldConfigVec.push_back(ASConfig);

	// Curtiss single
	CSConfig.name = CSP;
	CSConfig.rangeStart = 1;
	CSConfig.rangeEnd = 94;
	CSConfig.rowStart = 1;
	CSConfig.rowEnd = 13;
	CSConfig.id = 4;
	// rep1 SHORT_PLANT 1 
	SFR.setParam(1, 12, 1, 13);
	CSConfig.addTypeSubFieldRect(REP_1, SHORT_PLANT, SFR);
	// rep1 TALL_PLANT
	SFR.setParam(14, 25, 1, 13);
	CSConfig.addTypeSubFieldRect(REP_1, TALL_PLANT, SFR);
	// rep1 PS_PLANT
	SFR.setParam(27, 46, 1, 13);
	CSConfig.addTypeSubFieldRect(REP_1, PS_PLANT, SFR);
	// rep2 SHORT_PLANT 1 
	SFR.setParam(48, 59, 1, 13);
	CSConfig.addTypeSubFieldRect(REP_2, SHORT_PLANT, SFR);
	// rep2 TALL_PLANT
	SFR.setParam(61, 72, 1, 13);
	CSConfig.addTypeSubFieldRect(REP_2, TALL_PLANT, SFR);
	// rep2 PS_PLANT
	SFR.setParam(73, 94, 1, 13);
	CSConfig.addTypeSubFieldRect(REP_2, PS_PLANT, SFR);

	fieldConfigVec.push_back(CSConfig);

};

bool Map2FileMapping::checkRangeRowInBoundForType(int field, int type, int range, int row)
{
	if (fieldConfigVec[field].repVec[REP_1][type].size() != 0)
	{
		if (range >= fieldConfigVec[field].repVec[REP_1][type][0]._rangeStart &&
			range <= fieldConfigVec[field].repVec[REP_1][type][0]._rangeEnd &&
			row >= fieldConfigVec[field].repVec[REP_1][type][0]._rowStart &&
			row <= fieldConfigVec[field].repVec[REP_1][type][0]._rowEnd)
			return true;
	}

	if (fieldConfigVec[field].repVec[REP_2][type].size() != 0)
	{
		if (range >= fieldConfigVec[field].repVec[REP_2][type][0]._rangeStart &&
			range <= fieldConfigVec[field].repVec[REP_2][type][0]._rangeEnd &&
			row >= fieldConfigVec[field].repVec[REP_2][type][0]._rowStart &&
			row <= fieldConfigVec[field].repVec[REP_2][type][0]._rowEnd)
			return true;
	}

	return false;
}

void Map2FileMapping::setSideRow(int row, int* fileRow, int* side)
{
	int remainder = row % 4;

	if (remainder == 0 || remainder == 3)	// plant on the right side of vehicle
	{
		*side = RIGHT;
	}
	else if (remainder == 1 || remainder == 2) // plant on the left side of vehicle
	{
		*side = LEFT;
	}

	*fileRow = row / 2;
}


bool Map2FileMapping::getFileName(int field, int range, int row, std::vector<std::string>& pathVec, int& plantSide)
{
	int fileRow = -1;
	int fileRange = -1;
	int rep = -1;

	int side = -1;

	std::string path;

	pathVec.clear();

	if (field == AH || field == CH1 || field==CH2 )
	{
		if (field == AH)
		{
			if (row < 3 || row > 52 || range < 2 || range > 54)
			{
				std::cout << "row or range out of bound!" << std::endl;
				return false;
			}

			setSideRow(row, &fileRow, &side);

			fileRange = range;
			rep = 1;
		}
		else if (field == CH1 || field == CH2)
		{
			if (row < 3 || row > 92 || range < 2 || range > 39)
			{
				std::cout << "row or range out of bound!" << std::endl;
				return false;
			}

			if (row < 49)	// first part
			{
				setSideRow(row, &fileRow, &side);
			}
			if (row > 52)	// 2nd part
			{
				setSideRow(row - 50, &fileRow, &side);
			}
			else if (row < 53 && row > 48)
			{
				std::cout << "Filler row!" << std::endl;
				return false;
			}

			if (row > 52)
			{
				fileRow += 24;
			}

			fileRange = range;
			rep = 3;
		}

		if (side == LEFT)
			path = "Left_Re_";
		else
			path = "Right_Re_";

		//std::cout << "My Row:" << std::to_string(fileRow)<<std::endl;
		pathVec.push_back(path + std::to_string(rep) + "_Ro_" + std::to_string(fileRow) + "_Ra_" + std::to_string(fileRange) + "_Ca_");
		plantSide = side;
	}
	else if (field == AS || field == CSP)
	{
		if (field == AS)
		{
			if (row < 1 || row > 18 || range < 1 || range > 69)
			{
				std::cout << "row or range out of bound !" << std::endl;
				return false;
			}

			rep = 2;

		}
		else if (field == CSP)
		{
			if (row < 1 || row > 13 || range < 1 || range > 94)
			{
				std::cout << "row or range out of bound !" << std::endl;
				return false;
			}

			rep = 4;
		}

		// odd rows on right side of vehicle
		if (row % 2 == 1)
			side = RIGHT;
		else // even rows on left side of vehicle
			side = LEFT;

		fileRow = row;

		fileRange = range;

		if (side == LEFT)
			path = "Left_Re_";
		else
			path = "Right_Re_";

		pathVec.push_back(path + std::to_string(rep) + "_Ro_" + std::to_string(fileRow) + "_Ra_" + std::to_string(fileRange) + "_Ca_");
		pathVec.push_back(path + std::to_string(rep) + "_Ro_" + std::to_string(fileRow + 1) + "_Ra_" + std::to_string(fileRange) + "_Ca_");
		plantSide = side;
	}

	return true;
}

// number: number of images needed, plantLocation (range, row)
bool Map2FileMapping::getRandomImages(int field, int rep, int type, int number, std::vector<std::vector<int>>& plantLocationVec)
{

	// how many subfields
	int numSubField = fieldConfigVec[field].repVec[rep][type].size();

	float totalSum = 0;

	for (int i = 0; i < numSubField; i++)
	{
		totalSum += (float)fieldConfigVec[field].repVec[rep][type][i]._sum;
	}

	if (number > totalSum)
	{
		std::cout << "Number > total plants!" << std::endl;
		return false;
	}

	for (int i = 0; i < numSubField; i++)
	{
		SubFieldRect SFR = fieldConfigVec[field].repVec[rep][type][i];

		int subNum;

		if (i < numSubField - 1)
			subNum = roundf((float)SFR._sum / totalSum*number);
		else
			subNum = number;

		std::uniform_int_distribution<int> rangeDistri(SFR._rangeStart, SFR._rangeEnd);
		std::uniform_int_distribution<int> rowDistri(SFR._rowStart, SFR._rowEnd);

		// generate subNum random
		for (int j = 0; j < subNum; j++)
		{
			std::vector<int> plantLocation;

			plantLocation.push_back(rangeDistri(generator));
			plantLocation.push_back(rowDistri(generator));

			plantLocationVec.push_back(plantLocation);
		}

		number -= subNum;
		totalSum -= (float)fieldConfigVec[field].repVec[rep][type][i]._sum;
	}

	return true;
}