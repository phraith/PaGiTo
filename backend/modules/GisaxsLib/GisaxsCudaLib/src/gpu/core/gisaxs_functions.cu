#include "gpu/core/gisaxs_functions.h"
#include "device_launch_parameters.h"

#include "gpu/ff/sphere.h"
#include "gpu/ff/cylinder.h"
#include "gpu/ff/trapezoid.h"

#include <cstdio>

#include "standard_vector_types.h"
#include "common/standard_constants.h"
#include "gpu/util/cuda_numerics.h"
#include "common/flat_unitcell.h"


__constant__ DevUnitcell* dunitcell;

__global__ void update_unitcell(MyType2* params, int shape_count)
{
	int rvs_idx = 0;

	for (int i = 0; i < shape_count; ++i)
	{
		ShapeFF* shape = dunitcell->GetShape(i);

		switch (shape->Type())
		{
		case ShapeTypeV2::cylinder:
		{
			CylinderFF* cylinder = (CylinderFF*)shape;
			cylinder->Update( params[rvs_idx], params[rvs_idx + 1]);
			break;
		}
            case ShapeTypeV2::sphere:
		{
			SphereFF* sphere = (SphereFF*)shape;
			sphere->Update(params[rvs_idx]);
			break;
		}
//		case ShapeType::kTrapezoid:
//		{
//			TrapezoidFF* trapezoid = (TrapezoidFF*)shape;
//
//			for (int i = 0; i < trapezoid->BetaCount(); ++i)
//			{
//				trapezoid->UpdateBeta({ params[rvs_idx + 2 * i], params[rvs_idx + 2 * i + 1] }, i);
//			}
//			int next_idx = rvs_idx + 2 * trapezoid->BetaCount();
//
//			trapezoid->Update({ params[next_idx], params[next_idx + 1] }, { params[next_idx + 2], params[next_idx + 3] });
//			break;
//		}
		default:
			break;
		}

		rvs_idx += 2 * shape->ParamCount();
	}
}

__global__ void update_shapes(MyType* rands, int rand_count, int shape_count)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= shape_count * rand_count)
		return;

	int local_rand_id = tid % rand_count;
	int shape_idx = tid / rand_count;

	int first_param_idx = 0;

	ShapeFF* shape = dunitcell->GetShape(shape_idx);
	for (int i = 0; i < shape_idx; ++i)
	{
		first_param_idx += shape->ParamCount();
	}

	int first_rand_idx = first_param_idx * rand_count;

	switch (shape->Type())
	{
	case ShapeTypeV2::sphere:
	{
		SphereFF* sphere = (SphereFF*)shape;

		MyType2 radius = sphere->Radius();

		sphere->RandRads()[local_rand_id] = rands[first_rand_idx + local_rand_id] * radius.y + radius.x;
		break;
	}
	case ShapeTypeV2::cylinder:
	{
		CylinderFF* cylinder = (CylinderFF*)shape;

		MyType2 radius = cylinder->Radius();
		MyType2 height = cylinder->Height();

		cylinder->RandRads()[local_rand_id] = rands[first_rand_idx + local_rand_id] * radius.y + radius.x;
		cylinder->RandHeights()[local_rand_id] = rands[first_rand_idx + rand_count + local_rand_id] * height.y + height.x;
		break;
	}
//	case ShapeTypeV2::kTrapezoid:
//	{
//		TrapezoidFF* trapezoid = (TrapezoidFF*)shape;
//		MyType2 *beta = trapezoid->Beta();
//		MyType2 L = trapezoid->L();
//		MyType2 h = trapezoid->H();
//
//		for (int i = 0; i < trapezoid->BetaCount(); ++i)
//		{
//			trapezoid->RandBetas()[i * rand_count + local_rand_id] = rands[first_rand_idx + i * rand_count + local_rand_id] * beta[i].y + beta[i].x;
//		}
//
//		int next_idx = first_rand_idx + trapezoid->BetaCount() * rand_count;
//
//		trapezoid->RandLs()[local_rand_id] = rands[next_idx + local_rand_id] * L.y + L.x;
//		trapezoid->RandHs()[local_rand_id] = rands[next_idx + rand_count + local_rand_id] * h.y + h.x;
//		break;
//	}
	default:
		break;
	}
}

__device__ MyComplex EvalStructureFactor(MyComplex qx, MyComplex qy, MyComplex qz, float3 d, MyType n)
{
	MyComplex r = qx * d.x + qy * d.y + qz * d.z;
	if (r.x == 0 && r.y == 0)
		return { n, 0 };
	return (1. - cuCexpi(-1.f * r * n)) / (1. - cuCexpi(-1.f * r));
}


__global__ void cuda_run_gisaxs(MyType2* qpoints_xy, MyType2* qz, MyComplex4* coefficients, MyType* intensities, int shape_count, int qcount)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= qcount)
		return;

	MyComplex x = qpoints_xy[tid];
	MyComplex y = qpoints_xy[qcount + tid];

	MyComplex4 coeff = coefficients[tid];

	MyType intensity = 0;
	for (int i = 0; i < COHERENCY_DRAW_RATIO.y; ++i)
	{
		MyComplex scattering = { 0,0 };
		for (int j = 0; j < COHERENCY_DRAW_RATIO.x; ++j)
		{
			int current_iteration = i * COHERENCY_DRAW_RATIO.x + j;
			for (int k = 0; k < shape_count; ++k)
			{
				ShapeFF* shape = dunitcell->GetShape(k);
				scattering = scattering + coeff.x * shape->Evaluate(x, y, qz[tid], current_iteration);
				scattering = scattering + coeff.y * shape->Evaluate(x, y, qz[qcount + tid], current_iteration);
				scattering = scattering + coeff.z * shape->Evaluate(x, y, qz[2 * qcount + tid], current_iteration);
				scattering = scattering + coeff.w * shape->Evaluate(x, y, qz[3 * qcount + tid], current_iteration);
			}
		}
		intensity += scattering.x * scattering.x + scattering.y * scattering.y;
	}
	intensities[tid] = intensity;
	}
	
	__global__ void cuda_calc_sf(MyType2* qxy, MyType2* qz, MyComplex* sfs, int qcount)
	{
		int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= qcount)
			return;

		MyComplex qx = qxy[tid];
		MyComplex qy = qxy[qcount + tid];

		MyComplex qz1 = qz[tid];
		MyComplex qz2 = qz[qcount + tid];
		MyComplex qz3 = qz[2 * qcount + tid];
		MyComplex qz4 = qz[3 * qcount + tid];

		MyType3 d = dunitcell->Distances();
		MyType3I n = dunitcell->Repetitions();

		float3 dx = { d.x, 0, 0 };
		float3 dy = { 0, d.y, 0 };
		float3 dz = { 0, 0, d.z };


		sfs[tid] = EvalStructureFactor(qx, qy, qz1, dx, n.x)
			* EvalStructureFactor(qx, qy, qz1, dy, n.y)
			* EvalStructureFactor(qx, qy, qz1, dz, n.z);

		sfs[qcount + tid] = EvalStructureFactor(qx, qy, qz2, dx, n.x)
			* EvalStructureFactor(qx, qy, qz2, dy, n.y)
			* EvalStructureFactor(qx, qy, qz2, dz, n.z);

		sfs[2 * qcount + tid] = EvalStructureFactor(qx, qy, qz3, dx, n.x)
			* EvalStructureFactor(qx, qy, qz3, dy, n.y)
			* EvalStructureFactor(qx, qy, qz3, dz, n.z);

		sfs[3 * qcount + tid] = EvalStructureFactor(qx, qy, qz4, dx, n.x)
			* EvalStructureFactor(qx, qy, qz4, dy, n.y)
			* EvalStructureFactor(qx, qy, qz4, dz, n.z);
	}


	//__global__ calc_ff(MyComplex *ffus, MyComplex* qpar, MyComplex* qabs, MyType2* qz, int calculations, int shape_count, int qcount)
	//{
	//	//n*qcount*k*4
	//	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	//	if (tid >= calculations)
	//		return;

	//	int idx = tid / COHERENCY_DRAW_RATIO.y;
	//	int loc_n = tid % COHERENCY_DRAW_RATIO.y;

	//	for (int k = 0; k < shape_count; ++k)
	//	{
	//		ShapeFF* shape = dunitcell->GetShape(k);
	//		MyComplex qpar_idx = qpar[idx];
	//		for (int i = 0; i < 4; ++i)
	//		{
	//			MyComplex qz_c = qz[i * qcount + idx];
	//			MyComplex qabs_c = qabs[i * qcount + idx];

	//			MyComplex shape_sum = { 0,0 };
	//			for (int j = 0; j < COHERENCY_DRAW_RATIO.x; ++j)
	//			{
	//				int current_iteration = j * COHERENCY_DRAW_RATIO.y + loc_n;
	//				shape_sum = shape_sum + shape->Evaluate2(qpar_idx, qabs_c, qz_c, current_iteration);
	//			}

	//			ffus[qcount]
	//		}
	//	}
	//}


	__global__ void cuda_run_gisaxs_opt4(MyComplex* qpar, MyComplex* qabs, MyType2* qxy, MyType2* qz, int calculations, MyComplex* coefficients, MyType* intensities, int shape_count, int qcount, MyComplex* sfs)
	{
		int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= calculations)
			return;

		int idx = tid / COHERENCY_DRAW_RATIO.y;
		int loc_n = tid % COHERENCY_DRAW_RATIO.y;

		MyComplex qx = qxy[idx];
		MyComplex qy = qxy[qcount + idx];

		MyComplex qpar_idx = qpar[idx];

		MyType3* shape_locations = dunitcell->ShapeLocations();
		int* location_counts = dunitcell->LocationCounts();

		MyComplex scattering = { 0,0 };
		int cur_loc_idx = 0;
		for (int k = 0; k < shape_count; ++k)

		{

			int loc_count = location_counts[k];
			ShapeFF* shape = dunitcell->GetShape(k);

			for (int i = 0; i < 4; ++i)
			{
				MyComplex qz_c = qz[i * qcount + idx];
				MyComplex qabs_c = qabs[i * qcount + idx];
				MyComplex sfs_c = sfs[i * qcount + idx];

				MyComplex shape_sum = { 0,0 };
				for (int j = 0; j < COHERENCY_DRAW_RATIO.x; ++j)
				{
					int current_iteration = j * COHERENCY_DRAW_RATIO.y + loc_n;
					shape_sum = shape_sum + shape->Evaluate2(qpar_idx, qabs_c, qz_c, current_iteration);
					
				}
				MyComplex shape_sum_u = { 0,0 };
				for (int l = 0; l < loc_count; ++l)
				{
					MyType3 loc = shape_locations[cur_loc_idx + l];
					MyComplex qr = qx * loc.x + qy * loc.y + qz_c * loc.z;

					shape_sum_u = shape_sum_u + shape_sum *cuCexpi(-1.f * qr);
				}

				scattering = scattering + coefficients[i * qcount + idx] * shape_sum_u *sfs_c;
			}
			cur_loc_idx += loc_count;
		}
		
		MyType scatter_abs = cuCabs(scattering);
		MyType intensity = scatter_abs * scatter_abs;
		if (!isnan(intensity))
			atomicAdd(&intensities[idx], intensity);
	}


	//__global__ void cuda_run_gisaxs_opt3(MyComplex* qpar, MyComplex* qabs, MyType2* qxy, MyType2* qz, int calculations, MyComplex* coefficients, MyType* intensities, int shape_count, int qcount, MyComplex* sfs)
	//{
	//	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	//	if (tid >= calculations)
	//		return;

	//	int idx = tid / COHERENCY_DRAW_RATIO.y;
	//	int loc_n = tid % COHERENCY_DRAW_RATIO.y;

	//	MyComplex qx = qxy[idx];
	//	MyComplex qy = qxy[qcount + idx];

	//	MyComplex qpar_idx = qpar[idx];

	//	MyType3* shape_locations = dunitcell->ShapeLocations();
	//	int* location_counts = dunitcell->LocationCounts();

	//	MyComplex scattering = { 0,0 };
	//	int cur_loc_idx = 0;
	//	for (int k = 0; k < shape_count; ++k)
	//	{
	//		int loc_count = location_counts[k];
	//		ShapeFF* shape = dunitcell->GetShape(k);
	//		scattering = scattering + shape->Evaluate3(qpar_idx, qabs, qz, qcount, 
	//			idx, qx, qy, sfs, loc_n, loc_count, shape_locations, cur_loc_idx, coefficients);
	//		
	//		cur_loc_idx += loc_count;
	//	}

	//	MyType scatter_abs = cuCabs(scattering);
	//	MyType intensity = scatter_abs * scatter_abs;

	//	if (!isnan(intensity))
	//		atomicAdd(&intensities[idx], intensity);
	//}


	__global__ void cuda_rm_math_errors(MyType* intensities, int qcount)
	{
		int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= qcount)
			return;

		MyType intensity = intensities[tid];

		if (isnan(intensity) || isinf(intensity))
			intensities[tid] = 100;
	}

	__global__ void cuda_run_gisaxs_opt2(MyComplex* qpar, MyComplex* qabs, MyType2* qxy, MyType2* qz, int calculations, MyComplex* coefficients, MyType* intensities, int shape_count, int qcount, MyComplex *sfs)
	{
		int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= calculations)
			return;

		int idx = tid / COHERENCY_DRAW_RATIO.y;
		int loc_n = tid % COHERENCY_DRAW_RATIO.y;

		MyComplex scattering = { 0,0 };

		MyComplex qx = qxy[idx];
		MyComplex qy = qxy[qcount + idx];

		MyComplex qp = qpar[idx];

		MyType3* shape_locations = dunitcell->ShapeLocations();
		int* location_counts = dunitcell->LocationCounts();

		for (int i = 0; i < 4; ++i)
		{
			MyComplex qz_c = qz[i * qcount + idx];
			MyComplex sfs_c = sfs[i * qcount + idx];
			MyComplex qa = qabs[i * qcount + idx];

			for (int j = 0; j < COHERENCY_DRAW_RATIO.x; ++j)
			{
				int current_iteration = j * COHERENCY_DRAW_RATIO.y + loc_n;

				MyComplex ff_u = { 0,0 };

				int cur_loc_idx = 0;
				for (int k = 0; k < shape_count; ++k)
				{
					ShapeFF* shape = dunitcell->GetShape(k);
					MyComplex ff = shape->Evaluate2(qp, qa, qz_c, current_iteration);

					int loc_count = location_counts[k];
					for (int l = 0; l < loc_count; ++l)
					{
						MyType3 loc = shape_locations[cur_loc_idx + l];
						MyComplex qr = qx * loc.x + qy * loc.y + qz_c * loc.z;

						ff_u = ff_u + ff * cuCexpi(-1.f * qr);

					}
					cur_loc_idx += loc_count;
				}

				scattering = scattering + coefficients[i * qcount + idx] * ff_u * sfs_c;
			}
		}
		MyType scatter_abs = cuCabs(scattering);
		MyType intensity = scatter_abs  * scatter_abs;
		
		if (!isnan(intensity) && !isinf(intensity))
			atomicAdd(&intensities[idx], intensity);
	}


	__global__ void cuda_run_gisaxs_opt(MyComplex* qpar, MyComplex* qabs, MyType2* qxy, MyType2* qz, int calculations, MyComplex* coefficients, MyType* intensities, int shape_count, int qcount, MyComplex* sfs)
	{
		int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= calculations)
			return;

		int idx = tid / COHERENCY_DRAW_RATIO.y;
		int loc_n = tid % COHERENCY_DRAW_RATIO.y;

		MyComplex scattering = { 0,0 };

		MyComplex qx = qxy[idx];
		MyComplex qy = qxy[qcount + idx];

		MyType3* shape_locations = dunitcell->ShapeLocations();
		int* location_counts = dunitcell->LocationCounts();

		for (int i = 0; i < 4; ++i)
		{
			MyComplex qz_c = qz[i * qcount + idx];
			MyComplex sfs_c = sfs[i * qcount + idx];

			for (int j = 0; j < COHERENCY_DRAW_RATIO.x; ++j)
			{
				int current_iteration = j * COHERENCY_DRAW_RATIO.y + loc_n;

				MyComplex ff_u = { 0,0 };

				int cur_loc_idx = 0;
				for (int k = 0; k < shape_count; ++k)
				{
					ShapeFF* shape = dunitcell->GetShape(k);
					MyComplex ff = shape->Evaluate(qx, qy, qz_c, current_iteration);

					int loc_count = location_counts[k];
					for (int l = 0; l < loc_count; ++l)
					{
						MyType3 loc = shape_locations[cur_loc_idx + l];
						MyComplex qr = qx * loc.x + qy * loc.y + qz_c * loc.z;

						ff_u = ff_u + ff * cuCexpi(-1.f * qr);

					}
					cur_loc_idx += loc_count;
				}

				scattering = scattering + coefficients[i * qcount + idx] * ff_u * sfs_c;
			}
		}
		MyType scatter_abs = cuCabs(scattering);
		MyType intensity = scatter_abs * scatter_abs;

		if (!isnan(intensity) && !isinf(intensity))
			atomicAdd(&intensities[idx], intensity);
	}


	__device__ DevUnitcell::DevUnitcell(ShapeTypeV2 * shape_types, int shape_count, int rand_count, MyType3* shape_locations, int *locations_counts, MyType3I repetitions, MyType3 distances)
		:
		shapes_(new ShapeFF * [shape_count]),
		shape_count_(shape_count),
		shape_locations_(shape_locations),
		locations_counts_(locations_counts),
		repetitions_(repetitions),
		distances_(distances)
	{
		for (int i = 0; i < shape_count; ++i)
		{
			switch (shape_types[i])
			{
			case ShapeTypeV2::sphere:
			{
				shapes_[i] = new SphereFF({ -1, -1 }, rand_count);
				break;
			}
			case ShapeTypeV2::cylinder:
			{
				shapes_[i] = new CylinderFF({ -1, -1 }, { -1, -1 }, rand_count);
				break;
			}
//			case ShapeType::kTrapezoid:
//			{
//				shapes_[i] = new TrapezoidFF({ -1, -1 }, { -1, -1 }, { -1, -1 }, rand_count);
//				break;
//			}
			default:
				break;
			}
		}
	}

	__device__ DevUnitcell::~DevUnitcell()
	{
		for (int i = 0; i < shape_count_; ++i)
		{
			if (shapes_[i] != nullptr)
				delete shapes_[i];
		}
	}

	__device__ ShapeFF* DevUnitcell::GetShape(int idx)
	{
		if (idx < shape_count_)
			return shapes_[idx];

		return nullptr;
	}

	__device__ MyType3* DevUnitcell::ShapeLocations()
	{
		return shape_locations_;
	}

	__device__ int* DevUnitcell::LocationCounts()
	{
		return locations_counts_;
	}

	__device__ MyType3I DevUnitcell::Repetitions()
	{
		return repetitions_;
	}

	__device__ MyType3 DevUnitcell::Distances()
	{
		return distances_;
	}


	__global__ void create_unitcell(DevUnitcell * *dev_unitcell, ShapeTypeV2 * shape_types, int shape_count, MyType3* locations, int *locations_counts, int rand_count, MyType3I repetitions, MyType3 distances)
	{
		*dev_unitcell = new DevUnitcell(shape_types, shape_count, rand_count, locations, locations_counts, repetitions, distances);
	}

	__global__ void destroy_unitcell()
	{
		if (dunitcell != nullptr)
			delete dunitcell;
	}

	namespace Gisaxs
	{
		void CreateUnitcell(DevUnitcell** dev_unitcell, ShapeTypeV2* shape_types, int shape_count, MyType3 *locations, int *locations_counts, int rand_count, MyType3I repetitions, MyType3 distances, cudaStream_t work_stream)
		{

			create_unitcell << < 1, 1, 0, work_stream >> > (dev_unitcell, shape_types, shape_count, locations, locations_counts, rand_count, repetitions, distances);
			cudaMemcpyToSymbol(dunitcell, &(*dev_unitcell), sizeof(DevUnitcell *), 0, cudaMemcpyDeviceToDevice);
		}

		void DestroyUnitcell(cudaStream_t work_stream)
		{
			destroy_unitcell << < 1, 1, 0, work_stream >> > ();
		}

		void Update(MyType* rands, int qcount, MyType2* params, int shape_count, cudaStream_t work_stream)
		{
			int m = COHERENCY_DRAW_RATIO.x;
			int n = COHERENCY_DRAW_RATIO.y;

			update_unitcell << <1, 1, 0, work_stream >> > ( params, shape_count);
			update_shapes << <(shape_count * n * m) / RAND_THREADS + 1, RAND_THREADS, 0, work_stream >> > (rands, RANDOM_DRAWS, shape_count);
		}

		void RunSim(MyComplex* qpar, MyComplex* q, MyComplex* qpoints_xy, MyComplex* qpoints_z_coeffs, int qcount, MyComplex* coefficients, MyType* intensities, int shape_count, MyComplex *sfs, cudaStream_t work_stream)
		{
			int threads = 128;
			int calculations = qcount * COHERENCY_DRAW_RATIO.y;
			int blocks = calculations / threads + 1;
			//int blocks = 1024;

			// update_unitcell << <1, 1, 0, work_stream >> > (dev_unitcell, params, shape_count);
			// update_shapes << <(shape_count * n * m) / RAND_THREADS + 1, RAND_THREADS, 0, work_stream >> > (dev_unitcell, rands, RANDOM_DRAWS, shape_count);
			cuda_calc_sf << <  qcount / threads + 1, threads, 0, work_stream >> > (qpoints_xy, qpoints_z_coeffs, sfs, qcount);
			cuda_run_gisaxs_opt4 << <  blocks, threads, 0, work_stream >> > (qpar, q, qpoints_xy, qpoints_z_coeffs, calculations, coefficients, intensities, shape_count, qcount, sfs);
			//cuda_run_gisaxs_opt2 << <  blocks, threads, 0, work_stream >> > (qpar, q, qpoints_xy, qpoints_z_coeffs, calculations, coefficients, intensities, shape_count, qcount, sfs);
			
			//cuda_rm_math_errors << <  qcount / threads + 1, threads, 0, work_stream >> > (intensities, qcount);
			//cuda_run_gisaxs_opt << <  qcount / threads + 1, threads, 0, work_stream >> > (qpar, q, qpoints_xy, qpoints_z_coeffs, calculations, coefficients, intensities, shape_count, qcount, dev_unitcell, sfs);
			
			//cuda_run_gisaxs_opt << <  blocks, threads, 0, work_stream >> > (qpar, q, qpoints_xy, qpoints_z_coeffs, calculations, coefficients, intensities, shape_count, qcount, dev_unitcell);
			//cuda_run_gisaxs << <  blocks, threads, 0, work_stream >> > (qpoints_xy, qpoints_z_coeffs, coefficients, intensities, shape_count, qcount, dev_unitcell);
		}
	}