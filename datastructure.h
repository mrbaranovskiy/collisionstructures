#pragma once

#include "cuda_common.h"


struct Vector3d {
	double X; double Y; double Z;

	CALLABLE Vector3d operator+(const Vector3d& o) noexcept
	{
		const Vector3d vec{ X = this->X + o.X,Y = this->Y + o.Y,Z = this->Z + o.Z };

		return vec;
	}
};

struct Vector3f { float x, y, z; };

struct Matrix3x3d
{
	double E00;
	double E01;
	double E02;
	double E10;
	double E11;
	double E12;
	double E20;
	double E21;
	double E22;

	CALLABLE Matrix3x3d transpose() noexcept
	{
		const Matrix3x3d mtx{ E00 = E00,  E01 = E10,  E02 = E20, E10 = E01,
		  E11 = E11,  E12 = E21,  E20 = E02,  E21 = E12,  E22 = E22
		};

		return mtx;
	}
};

struct Matrix4x4d
{
	double E00;
	double E01;
	double E02;
	double E03;
	double E10;
	double E11;
	double E12;
	double E13;
	double E20;
	double E21;
	double E22;
	double E23;

	/*	CALLABLE explicit Matrix4x4d(Matrix3x3d linear) noexcept {
			E00 = linear.E00;
			E01 = linear.E01;
			E02 = linear.E02;
			E10 = linear.E10;
			E11 = linear.E11;
			E12 = linear.E12;
			E20 = linear.E20;
			E21 = linear.E21;
			E22 = linear.E22;
			E03 = 0.0;
			E13 = 0.0;
			E23 = 0.0;
		}

		CALLABLE explicit Matrix4x4d(Matrix3x3d mtx, Vector3d translation) noexcept : Matrix4x4d(mtx) {
			E03 = translation.X;
			E13 = translation.Y;
			E23 = translation.Z;
		}

		friend CALLABLE Matrix4x4d operator* (const Matrix4x4d& lhs, const Matrix4x4d& rhs) noexcept {

			Matrix4x4d mtx;

			mtx.E00 = lhs.E00 * rhs.E00 + lhs.E01 * rhs.E10 + lhs.E02 * rhs.E20;
			mtx.E01 = lhs.E00 * rhs.E01 + lhs.E01 * rhs.E11 + lhs.E02 * rhs.E21;
			mtx.E02 = lhs.E00 * rhs.E02 + lhs.E01 * rhs.E12 + lhs.E02 * rhs.E22;
			mtx.E03 = lhs.E00 * rhs.E03 + lhs.E01 * rhs.E13 + lhs.E02 * rhs.E23 + lhs.E03;
			mtx.E10 = lhs.E10 * rhs.E00 + lhs.E11 * rhs.E10 + lhs.E12 * rhs.E20;
			mtx.E11 = lhs.E10 * rhs.E01 + lhs.E11 * rhs.E11 + lhs.E12 * rhs.E21;
			mtx.E12 = lhs.E10 * rhs.E02 + lhs.E11 * rhs.E12 + lhs.E12 * rhs.E22;
			mtx.E13 = lhs.E10 * rhs.E03 + lhs.E11 * rhs.E13 + lhs.E12 * rhs.E23 + lhs.E13;
			mtx.E20 = lhs.E20 * rhs.E00 + lhs.E21 * rhs.E10 + lhs.E22 * rhs.E20;
			mtx.E21 = lhs.E20 * rhs.E01 + lhs.E21 * rhs.E11 + lhs.E22 * rhs.E21;
			mtx.E22 = lhs.E20 * rhs.E02 + lhs.E21 * rhs.E12 + lhs.E22 * rhs.E22;
			mtx.E23 = lhs.E20 * rhs.E03 + lhs.E21 * rhs.E13 + lhs.E22 * rhs.E23 + lhs.E23;

			return mtx;
		}

		CALLABLE Matrix3x3d getLinearPart() noexcept
		{
			Matrix3x3d mtx{ E00 = E00, E01 = E01, E02 = E02, E10 = E10, E11 = E11, E12 = E12, E20 = E20, E21 = E21, E22 = E22 };
			return mtx;
		}

		CALLABLE Vector3d transform(Vector3d v) noexcept;

		CALLABLE Matrix4x4d invertRigidMotion() noexcept
		{
			const auto linear = this->getLinearPart().transpose();
			Vector3d tr;

			tr.X = -linear.E00 * this->E03 - linear.E01 * this->E13 - linear.E02 * this->E23;
			tr.Y = -linear.E10 * this->E03 - linear.E11 * this->E13 - linear.E12 * this->E23;
			tr.Z = -linear.E20 * this->E03 - linear.E21 * this->E13 - linear.E22 * this->E23;

			Matrix4x4d inverted(linear, tr);

			return inverted;
		}*/

};

