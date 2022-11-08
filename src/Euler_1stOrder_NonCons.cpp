/*
Copyright 2021 W.R. Tonnon

This file is part of Semi-Lagrangian Tools
First-order solver for the incompressible Euler
equations using semi-Lagrangian advection of discrete differential forms.

Copyright (C) <2021>  <W.R. Tonnon>

Semi-Lagrangian Tools is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Semi-Lagrangian Tools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "mfem.hpp"
#include "Operators.h"
#include "Parameters.h"
#include "VertexValuedGridFunction.h"
#include <fstream>
#include <iostream>
#include <math.h>
#include <chrono>
#include <filesystem>
#include "SmallEdgeFiniteElement.h"


#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>


#define convergence_analysis 0 // 0 is no convergence analysis, 1 is convergence in dt and h, 2 is convergence in h only, 3 is convergence in dt only
//#define test_case 25// 0 is rotating - 1 is linear propagation - 5 3D
#define implementation_number 0

int main(int argc, char *argv[])
{

    // Parse the input parameters
    double timestep, par;
    int refinements, problem_number, prescribed_velocity, interpolation, visualisation;
    std::string mesh_file, output_file, config_file;

    // Declare the supported options
    boost::program_options::options_description cmd("command-line-only options");
    cmd.add_options()
            ("help,h", "produce help message")
            ("config-file,c", boost::program_options::value<std::string>(&config_file), "absolute path to configuration file")
            ;

    boost::program_options::options_description config("config (file or cmd) options");
    config.add_options()
            ("visualisation,v", boost::program_options::value<int>(&visualisation)->implicit_value(0)->default_value(-1), "enable visualisation, if argument is given:\n  <0 - visualisation off\n =0 - server-based visualisation\n >0 - file-based visualisation")
            ("timestep,t", boost::program_options::value<double>(&timestep), "timestep")
            ("parameter,par", boost::program_options::value<double>(&par), "parameter")
            ("refinements,r", boost::program_options::value<int>(&refinements), "number of mesh refinements")
            ("mesh-file,m", boost::program_options::value<std::string>(&mesh_file), "absolute path to the mesh file")
            ("problem-number,p", boost::program_options::value<int>(&problem_number), "problem number")
            ("prescribed-velocity,V", boost::program_options::value<int>(&prescribed_velocity), "")
            ("interpolation,i", boost::program_options::value<int>(&interpolation), "used interpolation:\n0 - angle-based averaging\n1 - cell-based averaging\n2 - line-based averaging\n3 - Use contniuous velocity")
            ("output-file-name,o", boost::program_options::value<std::string>(&output_file), "name of the output file without extension")
            ;

    boost::program_options::options_description all_options;
    all_options.add(cmd).add(config);

    // Read options from the command line
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc,argv,all_options), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << all_options << "\n";
        return 1;
    }

    // Read options from the config file if given
    if(vm.count("config-file")){
        std::ifstream config_stream(config_file.c_str());
        boost::program_options::store(boost::program_options::parse_config_file(config_stream,all_options), vm);
    }
    boost::program_options::notify(vm);

    // Check if all required options are given
    for(auto option : config.options())
    {
        if(!vm.count(option->canonical_display_name())) {
            std::cout << all_options << std::endl;
            throw std::runtime_error(option->canonical_display_name() + " required, but not given in either the command line or configuration file");
            return 1;
        }
    }


    /// Find the project root
    std::string project_root;
    char tmp[256];
    getcwd(tmp, 256);
    project_root = tmp;
    project_root += "/../../";
    size_t pos = project_root.find("semi-lagrangian-tools/")+std::string("semi-lagrangian-tools/").length(); //find location of word
    project_root.erase(pos,project_root.length()); //delete everything prior to location found

    // Problem-specific Parameters
    PhysicalParameters param(problem_number, par);
    mesh_file = param.GetMeshFile();
    double t_end = param.GetTfinal();
    auto velocity = [&param](mfem::Vector x, double t) -> mfem::Vector {
        return param.velocity(x, t);
    };

    // Read and prepare the mesh
    mfem::Mesh mesh(mesh_file.c_str(),1,1);
    mesh.ReorientTetMesh();
    int dim = mesh.Dimension();

    // Set some parameters
    double tol =1e-12;
    int order = 1;

    /// Prepare output file
    std::string output_folder = output_file;
    pos = output_folder.find("_r_"); //find location of word
    output_folder.erase(pos,output_folder.length()); //delete everything prior to location found

    std::ofstream output_stream;
    output_stream.open((project_root+"data/output/"+output_folder+"/"+output_file+".csv").c_str());
    output_stream<<"runtime [s],timestep [s],mesh-width [h],time [s],L2 Error u,Linf Error u,L2 Norm u,Helicity u" << std::endl;

    /// Prepare visualisation directory
    if(visualisation>0) {
        boost::filesystem::remove_all(project_root + "data/visualisation/" + output_file);
        boost::filesystem::create_directory(project_root + "data/visualisation/" + output_file);
    }

    /// Prepare output mesh file (to store refined mesh)
    std::string mesh_save_file = project_root+"data/mesh/"+output_file+".mesh";

    // Refine the mesh as required
    for(int i=0;i<refinements;++i)
        mesh.UniformRefinement();
    mesh.ReorientTetMesh();

    // Store the refined mesh for later reference
    mesh.Save(mesh_save_file.c_str());

    // Get some mesh characteristics
    double h_min, h_max, kappa_min, kappa_max;
    mesh.GetCharacteristics(h_min,h_max,kappa_min,kappa_max);

    // Define the Finite Element Collection and Spaces
    mfem::FiniteElementCollection *fec;
    fec = new WE_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_nodal = new mfem::H1_FECollection(order,dim);
    mfem::FiniteElementSpace *fes = new mfem::FiniteElementSpace(&mesh, fec);
    mfem::FiniteElementSpace *fes_nodal = new mfem::FiniteElementSpace(&mesh, fec_nodal);

    // Define unknowns
    VectorTraceBackGridFunction u_gf(fes);    // velocity at t
    mfem::GridFunction p_gf(fes_nodal);       // Pressure/Lagrange multiplier for incompressible Euler

    // Define the initial data
    auto InitialData_u = [&param](const mfem::Vector& x, mfem::Vector& out) -> void {
        out = param.VectorInitialData(x);
    };

    // Transform initial data to mfem::VectorFUnctionCoefficient type
    mfem::VectorFunctionCoefficient u0(param.GetVectorDim(),InitialData_u);

    // Project the initial data onto the GridFunctions.
    if(param.exactInitialConditionAvailable()){
        std::cout << "Exact initial conditions used\n";
        if(dim==2) u_gf.ProjectLeastSquaresCoefficient<mfem::Geometry::TRIANGLE>(InitialData_u);
        else u_gf.ProjectLeastSquaresCoefficient<mfem::Geometry::TETRAHEDRON>(InitialData_u);
    }
    else{
        std::cout << "Discrete initial conditions used\n";
        param.computeInitialCondition(u_gf);
    }


    // Initialize with 0 to prevent issues with NaN.
    p_gf = 0.;

    // We define a zero function as we will need this often
    mfem::Vector null(param.GetVectorDim());
    null = 0.;
    mfem::VectorConstantCoefficient zero(null);

    // Initialize timestep:
    // * if the given timestep is bigger than 0, we use the given timestep
    // * if the given timestep is smaller than 0, we use a CFL condition of .1 (based on the initial velocity)
    double dt;
    if(timestep>0) dt = timestep;
    else dt = param.GetTfinal()/std::max(ceil(param.GetTfinal()/(.1*h_min/u_gf.ComputeMaxError(zero))),1.);

    /// timestepping
    if(abs(round(t_end/dt)-t_end/dt)>0.0000000001 & dt<t_end)
        mfem::mfem_error("main: t_end needs to be an integer multiple of dt.");

    // We initialize a GridFunction that is supported by GLVis
    mfem::FiniteElementCollection *fec_vis  = new mfem::ND_FECollection(order, dim);
    mfem::FiniteElementSpace *fes_vis = new mfem::FiniteElementSpace(&mesh, fec_vis);
    mfem::GridFunction u_gf_vis(fes_vis);

    // We project u_gf onto the supported GridFunction
    mfem::VectorGridFunctionCoefficient u_gf_coeff(&u_gf);
    u_gf_vis.ProjectCoefficient(u_gf_coeff);

    // Initialize the visualisation: We generate a GLVis script and associated '.gf'-files to use GLVis for later visualisation.
    std::string visualisation_file = project_root + "data/visualisation/" + output_file +"/"+ output_file + "_it_000000.gf";
    std::string jpg_file = project_root + "data/visualisation/" + output_file +"/"+ output_file + "_it_000000.jpg";
    std::ofstream glvis_script;
    std::ofstream visualisation_stream;
    if(visualisation>0) {
        glvis_script.open((project_root+"data/visualisation/"+output_file+"/" +output_file+".glvs"));
        glvis_script << "window 0 0 300 300\n\n";
        glvis_script << "solution " + mesh_save_file + " " + visualisation_file << " screenshot " << jpg_file << std::endl;

        visualisation_stream.open(visualisation_file);
        u_gf_vis.Save(visualisation_stream);
        visualisation_stream.close();
    }

    /// Prepare Matrices that can be pre-computed
    // Find the right elements to change for the essential BCs
    mfem::Array<int> ess_tdof_list, ess_tdof_list_nodal;
    if (mesh.bdr_attributes.Size()) {
        mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
        ess_bdr = 0; // Here we do want to enforce Dirichlet boundary conditions
        fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
        fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list_nodal);
    }

    // a(u,v) = (u,v)_L2, u trial, v test
    mfem::BilinearForm a(fes), d(fes_nodal);
    mfem::SparseMatrix A, A_backup;
    a.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    a.Assemble();
    a.FormSystemMatrix(ess_tdof_list,A);
    a.FormSystemMatrix(ess_tdof_list,A_backup);

    // b(u,v) = (du,v)_L2, u trial, v test
    mfem::MixedBilinearForm b(fes_nodal,fes);
    mfem::SparseMatrix B;
    b.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
    b.Assemble();
    b.FormRectangularSystemMatrix(ess_tdof_list,ess_tdof_list_nodal,B);

    // c(u,v) = (u,dv)_L2, u trial, v test
    mfem::MixedBilinearForm c(fes,fes_nodal);
    mfem::SparseMatrix C;
    mfem::ConstantCoefficient minus_one(-1.);
    c.AddDomainIntegrator(new mfem::MixedVectorWeakDivergenceIntegrator(minus_one));
    c.Assemble();
    c.FormRectangularSystemMatrix(ess_tdof_list,ess_tdof_list_nodal,C);

    // d(u,v) = eps*(du,dv)_L2 for 1-forms, u trial, v test
    mfem::BilinearForm geps(fes);
    mfem::SparseMatrix Geps;
    mfem::ConstantCoefficient eps(param.GetEpsilon());
    geps.AddDomainIntegrator(new mfem::CurlCurlIntegrator(eps));
    geps.Assemble();
    geps.FormSystemMatrix(ess_tdof_list,Geps);
    // Prepare system matrix for both u and A
    mfem::SparseMatrix MHDSystem_u(A.NumRows() + C.NumRows());
    int size_system_u = A.NumRows() + C.NumRows();
    for (int k = 0; k < A.Size(); ++k) {
        mfem::Array<int> cols;
        mfem::Vector srow;

        A.GetRow(k, cols, srow);
        for (int q = 0; q < cols.Size(); ++q) {
            MHDSystem_u.Add(k, cols[q], srow[q]);
        }

        cols.DeleteAll();
        Geps.GetRow(k, cols, srow);
        for (int q = 0; q < cols.Size(); ++q) {
            MHDSystem_u.Add(k, cols[q], dt * srow[q]);
        }

    }
    for (int k = 0; k < B.NumRows(); ++k) {
        mfem::Array<int> cols;
        mfem::Vector srow;

        B.GetRow(k, cols, srow);
        for (int q = 0; q < cols.Size(); ++q) {
            MHDSystem_u.Add(k, A.NumCols() + cols[q], dt * srow[q]);
        }
    }
    for (int k = 0; k < C.NumRows(); ++k) {
        mfem::Array<int> cols;
        mfem::Vector srow;

        C.GetRow(k, cols, srow);
        for (int q = 0; q < cols.Size(); ++q){
            MHDSystem_u.Add(A.NumRows() + k, cols[q], dt * srow[q]);
        }
    }
    MHDSystem_u.Finalize();

    // Matrix required for postprocessing calculation of the helicity
    mfem::BilinearForm h(fes);
    mfem::SparseMatrix H;
    h.AddDomainIntegrator(new mfem::MixedVectorWeakCurlIntegrator());
    h.Assemble();
    h.FormSystemMatrix(ess_tdof_list,H);

    // Save the start time to estimate remaining runtime later
    auto start = std::chrono::system_clock::now();

    // Initialize timer
    auto start_loop = std::chrono::system_clock::now();

    // Loop over the timeloop
    int iteration=0;
    double t = 0.;
    bool finished = false;
    while (t<t_end-0.000000001) {
        // Progress bar and time estimation
        auto stop = std::chrono::system_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        float progress = (t)/param.GetTfinal();
        if (progress < 1.0) {
            int barWidth = 70;

            std::cout << "[";
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            double seconds_left = progress <1e-20? 0. : (1./progress-1.)*double(diff.count());
            double minutes_left = floor(seconds_left/60);
            seconds_left = seconds_left-minutes_left*60;
            if(progress>1e-20) std::cout << "] " << int(progress * 100.0) << "%, time remaining: " << minutes_left << " minutes, " << seconds_left << " seconds " <<" \r";
            else std::cout << "] " << int(progress * 100.0) << "% \r";
            std::cout.flush();
        }

        // Increase the time
        t += dt;

        // Prepare source data u
        auto source_u = [t, &param](const mfem::Vector& x, mfem::Vector& out)->void {
            out = param.VectorSourceData(x,t);
        };
        mfem::LinearForm flf(fes);
        mfem::VectorFunctionCoefficient source_u_coeff(param.GetVectorDim(), source_u);
        flf.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(source_u_coeff));
        flf.Assemble();

        // Reparametrize the exact solutions of u and A and their curls for enforcing boundary conditions and computing errors
        auto u_at_t = [&param,t](const mfem::Vector& x, mfem::Vector& out) -> void {
            out = param.VectorExactSolution(x,t);
            if(isnan(out.Elem(0))) out = 0.;
        };
        auto curl_u_at_t = [t, &param](const mfem::Vector &x, mfem::Vector &out) {
            out = param.VectorCurl(x, t);
        };

        // Convert exact functions for u and A to mfem::VectorFunctionCoefficient type
        mfem::VectorFunctionCoefficient u_at_t_coeff(param.GetVectorDim(), u_at_t);
        mfem::VectorFunctionCoefficient curl_u_at_t_coeff(param.GetVectorDim(), curl_u_at_t);

        // Prepare the normal boundary conditions
        mfem::LinearForm u_normal_bc(fes_nodal);
        u_normal_bc.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(u_at_t_coeff));
        u_normal_bc.Assemble();

        mfem::LinearForm curl_u_tangent_bc(fes);
        if(param.GetEpsilon()>0){
            curl_u_tangent_bc.AddBoundaryIntegrator(new mfem::VectorFEBoundaryTangentLFIntegrator(curl_u_at_t_coeff));
            curl_u_tangent_bc.Assemble();
        }
        else{
            curl_u_tangent_bc = 0.;
        }

        // We need the following for boundary conditions or for using when the option for prescribed velocities is set.
        std::function<void(const mfem::Vector &, mfem::Vector &)> velocity_at_t(
                [velocity, t, dt](const mfem::Vector &x, mfem::Vector &out) -> void {
                    out = velocity(x, t);
                });

        // To estimate the velocity at time t, we use a linear extrapolation
        PointWiseGridFunction *discrete_velocity_at_t = NULL;
        if (interpolation == 2) discrete_velocity_at_t = new SmoothenedPointWiseGridFunction(fes, velocity_at_t, param,&mesh);
        if (interpolation == 3) discrete_velocity_at_t = new RandomPointWiseGridFunction(fes, velocity_at_t, param, &mesh);
        discrete_velocity_at_t->Set(1., u_gf);


        std::function<void(const int, const mfem::Vector &, mfem::Vector &)> traceback_mapping_m1(
                [dim, dis_vel_at_t(discrete_velocity_at_t), vel_at_t(velocity_at_t), timestep(dt),prescribed_velocity](
                          const int elem,
                          const mfem::Vector &x,
                          mfem::Vector &out
                        ) -> void {
                              // We use Heun's method to achieve second-order timestepping
                              double tol = 1e-10;

                              // Compute the velocity at t: v_star = v(x,t)
                              mfem::Vector save, vel_star(dim);
                              if(!prescribed_velocity) vel_star = dis_vel_at_t->evalPointWise(x, elem, tol);
                              else vel_at_t(x,vel_star);

                              // Compute out = x - dt * v_star
                              out = x;
                              out.Add(-timestep, vel_star);

                              // Return
                              return;
                      });

        // Prepare boundary conditions for trace back functions
        /// TODO: make time-independent
        auto u_exact = [&param](const mfem::Vector& x, const double t, mfem::Vector& out) -> void {
            out = param.VectorExactSolution(x,t);
            if(isnan(out.Elem(0))) out = 0.;
        };

        // Compute X_{-2*dt}^* u_gf
        VectorTraceBackGridFunction u_prev(fes,
                                           traceback_mapping_m1,
                                           u_exact);
        if(dim==3) u_prev.TraceBackAndProjectGridFunctionPerElementSmallEdges<mfem::Geometry::TETRAHEDRON>(u_gf,t,dt);
        else if(dim==2) u_prev.TraceBackAndProjectGridFunctionPerElementSmallEdges<mfem::Geometry::TRIANGLE>(u_gf,t,dt);

        // We need to multiply with test functions
        mfem::Vector A_u_prev(u_prev.Size()), A_u_prev_prev(u_prev.Size());
        A.Mult(u_prev, A_u_prev);

        // It remains to compute the velocity, u
        // We start by defining the RHS of the system
        mfem::Vector RHS_u(size_system_u);
        RHS_u = 0.;

        for (int k = 0; k < u_prev.Size(); ++k) {
                RHS_u.Elem(k) = A_u_prev.Elem(k) + dt * flf.Elem(k) - dt * param.GetEpsilon() * curl_u_tangent_bc.Elem(k);
        }
        for (int k = u_gf.Size(); k < size_system_u; ++k)
                RHS_u.Elem(k) = dt * u_normal_bc.Elem(k - u_gf.Size());

        // Solve the system for u and p
        mfem::Vector sol_u(size_system_u);
        sol_u = 0.;
        mfem::MINRES(MHDSystem_u, RHS_u, sol_u, 0, 200000, tol * tol, tol * tol);

        // Check if the iteration converged sufficiently
        mfem::Vector residual(sol_u.Size());
        MHDSystem_u.Mult(sol_u, residual);
        residual -= RHS_u;
        if (!(residual.Norml2() < 10000 * tol)) {
                std::cout << "MINRES error u " << residual.Norml2() << std::endl;
                mfem::mfem_warning("Euler_1stOrder_NonCons(): MINRES did not converge.");
        }

        // Extract u and p from the vector
        VectorTraceBackGridFunction u_gf_temp(u_gf);
        mfem::Array<int> dofs;
        dofs.DeleteAll();
        for (int k = 0; k < A.NumRows(); ++k) dofs.Append(k);
        sol_u.GetSubVector(dofs, u_gf_temp);

        dofs.DeleteAll();
        for (int k = u_gf.Size(); k < size_system_u; ++k) dofs.Append(k);
        sol_u.GetSubVector(dofs, p_gf);

        // Set the values for the next iteration
        u_gf = u_gf_temp;

        // Clean up
        delete discrete_velocity_at_t;

        // increase the iteration count
        iteration++;

        /// Compute the total runtime up to this point (excluding preprocessing)
        auto stop_it = std::chrono::system_clock::now();
        auto diff_mu_s = std::chrono::duration_cast<std::chrono::microseconds>(stop_it - start);

        // Compute the quantities of interest
        double err_Linf_u = u_gf.ComputeMaxError(u_at_t_coeff);
        double err_L2_u   = u_gf.ComputeL2Error(u_at_t_coeff);
        double helicity_u = H.InnerProduct(u_gf,u_gf);
        double norm_L2_u  = std::sqrt(A.InnerProduct(u_gf,u_gf));

        // Save all the quantities of interest
        output_stream << diff_mu_s.count()  << ","
                      << dt                 <<","
                      << h_max              <<","
                      << t                  <<","
                      << err_L2_u           <<","
                      << err_Linf_u         <<","
                      << norm_L2_u          <<","
                      << helicity_u         <<std::endl;

        // On the last iteration print some information to std::cout
        if(t > t_end-1e-8){
            // Compute total time in human-readable quantities
            auto stop_loop = std::chrono::system_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::microseconds>(stop_loop - start);
            double seconds = diff.count()/1000;
            double minutes = floor(seconds/60000);
            seconds = seconds/1000 - minutes*60;

            // Print quantities of interest
            std::cout     << "runtime = "      << minutes    << " minutes, "
                                               << seconds    << " seconds, "
                          << "dt = "           << dt         << ", "
                          << "mesh_width = "   << h_max      << ", "
                          << "L2 Error u = "   << err_L2_u   << ", "
                          << "Linf Error u = " << err_Linf_u << std::endl;
        }

        // Project the gridfunction onto a standard FESpace for visualisation
        u_gf_coeff.SetGridFunction(&u_gf);
        u_gf_vis.ProjectCoefficient(u_gf_coeff);

        // We write out the data every 'visualisation' timesteps if visualisation is bigger than zero
        if(visualisation>0 & iteration%std::max(visualisation,1)==0) {
            // Convert iteration from 'int' to 'str'
            std::string index = std::to_string(iteration);

            // Pad with zeros to reach 6 digits if index is less than 6 digits long
            if(index.length()<6) index = std::string(6-index.length(), '0') + index;

            // Define the proper file paths
            visualisation_file = project_root + "data/visualisation/" + output_file +"/"+ output_file + "_it_" + index + ".gf";
            std::string jpg_file = project_root + "data/visualisation/" + output_file +"/"+ output_file + "_it_" + index + ".jpg";

            // Save the gridfunction DoFs
            u_gf_vis.Save(visualisation_file.c_str());

            // Write to GLVis script
            glvis_script << "solution " + mesh_save_file + " " + visualisation_file << " screenshot " << jpg_file << std::endl;

            // Save mfem meshes using VisitDataCollection
            mfem::VisItDataCollection dc("mesh_save_file", &mesh);
            dc.SetPrefixPath("");
            dc.RegisterField("velocity", &u_gf_vis);
            dc.Save();

            // Save meshes and grid functions in VTK format
            std::fstream vtkFs( project_root + "data/visualisation/" + output_file +"/"+ output_file + "_it_" + index + ".vtk" , std::ios::out);
            const int ref = 0;
            mesh.PrintVTK( vtkFs, ref);
            u_gf_vis.SaveVTK( vtkFs, "scalar_gf", ref);
        }



    }

    // Close the streams
    output_stream.close();
    glvis_script.close();

    // Clean up
    delete fes;
    delete fes_nodal;
    delete fec;
    delete fec_nodal;

    // Return
    return 0;
}
