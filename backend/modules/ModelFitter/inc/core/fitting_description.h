//#ifndef MODEL_FITTER_CORE_FITTING_DESCRIPTION_H
//#define MODEL_FITTER_CORE_FITTING_DESCRIPTION_H
//
//#include <memory>
//
//#include <common/experimental_model.h>
//
//class FitJob
//{
//public:
//	FitJob(std::shared_ptr<ExperimentalModel> model, std::unique_ptr<const ImageData> real_img, std::shared_ptr<Unitcell> h_unitcell,  std::string uuid, int evolutions, int individuals, int populations, bool is_last);
//	std::shared_ptr<ExperimentalModel> GetModel() const;
//
//	const ImageData* RealImg() const;
//	std::shared_ptr<Unitcell> HUnitcell() const;
//	const std::string& Uuid();
//	size_t Evolutions() const;
//	size_t Individuals() const;
//	size_t Populations() const;
//
//	bool IsLast();
//
//private:
//	std::shared_ptr<ExperimentalModel> model_;
//	std::unique_ptr<const ImageData> real_img_;
//	std::shared_ptr<Unitcell> h_unitcell_;
//
//	std::string uuid_;
//
//	int evolutions_;
//	int individuals_;
//	int populations_;
//
//	bool is_last_;
//};
//
//#endif