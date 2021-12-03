/*!
 * \file CAdjFlowOutput.hpp
 * \brief Headers of the adjoint flow output.
 * \author T. Kattmann
 * \version 7.2.1 "Blackbird"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2021, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "COutput.hpp"

/*! \class CAdjFlowOutput
 *  \brief Output class for flow discrete adjoint problems.
 *  \author T. Kattmann
 *  \date December 3, 2021.
 */
class CAdjFlowOutput : public COutput {
 protected:
  TURB_MODEL turb_model; /*!< \brief The kind of turbulence model*/
  bool cont_adj;         /*!< \brief Boolean indicating whether we run a cont. adjoint problem */
  bool frozen_visc;      /*!< \brief Boolean indicating whether frozen viscosity/turbulence is used. */

 public:
  /*!
   * \brief Constructor of the class
   * \param[in] config - Definition of the particular problem.
   */
  CAdjFlowOutput(CConfig* config, unsigned short nDim);

  /*!
   * \brief Add scalar (turbulence/species) history fields for the Residual RMS (FVMComp, FVMInc, FVMNEMO).
   */
  void AddHistoryOutputFields_AdjScalarRMS_RES(const CConfig* config);

  /*!
   * \brief Add scalar (turbulence/species) history fields for the max Residual (FVMComp, FVMInc, FVMNEMO).
   */
  void AddHistoryOutputFields_AdjScalarMAX_RES(const CConfig* config);

  /*!
   * \brief Add scalar (turbulence/species) history fields for the BGS Residual (FVMComp, FVMInc, FVMNEMO).
   */
  void AddHistoryOutputFields_AdjScalarBGS_RES(const CConfig* config);

  /*!
   * \brief Add scalar (turbulence/species) history fields for the linear solver (FVMComp, FVMInc, FVMNEMO).
   */
  void AddHistoryOutputFields_AdjScalarLinsol(const CConfig* config);

  /*!
   * \brief Set all scalar (turbulence/species) history field values.
   */
  void LoadHistoryData_AdjScalar(const CConfig* config, const CSolver* const* solver);

  /*!
   * \brief Add scalar (turbulence/species) volume solution fields for a point (FVMComp, FVMInc, FVMNEMO).
   * \note The order of fields in restart files is fixed. Therefore the split-up.
   * \param[in] config - Definition of the particular problem.
   */
  void SetVolumeOutputFields_AdjScalarSolution(const CConfig* config);

  /*!
   * \brief Add scalar (turbulence/species) volume solution fields for a point (FVMComp, FVMInc, FVMNEMO).
   * \note The order of fields in restart files is fixed. Therefore the split-up.
   * \param[in] config - Definition of the particular problem.
   */
  void SetVolumeOutputFields_AdjScalarResidual(const CConfig* config);

  /*!
   * \brief Set all scalar (turbulence/species) volume field values for a point.
   * \param[in] config - Definition of the particular problem.
   * \param[in] solver - The container holding all solution data.
   * \param[in] iPoint - Index of the point.
   */
  void LoadVolumeData_AdjScalar(const CConfig* config, const CSolver* const* solver, const unsigned long iPoint);
};
