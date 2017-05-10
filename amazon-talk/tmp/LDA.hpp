// Code generated by Stan version 2.14

#include <stan/model/model_header.hpp>

namespace LDA_model_namespace {

using std::istream;
using std::string;
using std::stringstream;
using std::vector;
using stan::io::dump;
using stan::math::lgamma;
using stan::model::prob_grad;
using namespace stan::math;

typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_d;
typedef Eigen::Matrix<double,1,Eigen::Dynamic> row_vector_d;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_d;

static int current_statement_begin__;

class LDA_model : public prob_grad {
private:
    int K;
    int V;
    int M;
    int N;
    vector<int> w;
    vector<int> doc;
    vector_d alpha;
    vector_d beta;
public:
    LDA_model(stan::io::var_context& context__,
        std::ostream* pstream__ = 0)
        : prob_grad(0) {
        typedef boost::ecuyer1988 rng_t;
        rng_t base_rng(0);  // 0 seed default
        ctor_body(context__, base_rng, pstream__);
    }

    template <class RNG>
    LDA_model(stan::io::var_context& context__,
        RNG& base_rng__,
        std::ostream* pstream__ = 0)
        : prob_grad(0) {
        ctor_body(context__, base_rng__, pstream__);
    }

    template <class RNG>
    void ctor_body(stan::io::var_context& context__,
                   RNG& base_rng__,
                   std::ostream* pstream__) {
        current_statement_begin__ = -1;

        static const char* function__ = "LDA_model_namespace::LDA_model";
        (void) function__; // dummy call to supress warning
        size_t pos__;
        (void) pos__; // dummy call to supress warning
        std::vector<int> vals_i__;
        std::vector<double> vals_r__;
        double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // suppress unused var warning

        // initialize member variables
        context__.validate_dims("data initialization", "K", "int", context__.to_vec());
        K = int(0);
        vals_i__ = context__.vals_i("K");
        pos__ = 0;
        K = vals_i__[pos__++];
        context__.validate_dims("data initialization", "V", "int", context__.to_vec());
        V = int(0);
        vals_i__ = context__.vals_i("V");
        pos__ = 0;
        V = vals_i__[pos__++];
        context__.validate_dims("data initialization", "M", "int", context__.to_vec());
        M = int(0);
        vals_i__ = context__.vals_i("M");
        pos__ = 0;
        M = vals_i__[pos__++];
        context__.validate_dims("data initialization", "N", "int", context__.to_vec());
        N = int(0);
        vals_i__ = context__.vals_i("N");
        pos__ = 0;
        N = vals_i__[pos__++];
        context__.validate_dims("data initialization", "w", "int", context__.to_vec(N));
        validate_non_negative_index("w", "N", N);
        w = std::vector<int>(N,int(0));
        vals_i__ = context__.vals_i("w");
        pos__ = 0;
        size_t w_limit_0__ = N;
        for (size_t i_0__ = 0; i_0__ < w_limit_0__; ++i_0__) {
            w[i_0__] = vals_i__[pos__++];
        }
        context__.validate_dims("data initialization", "doc", "int", context__.to_vec(N));
        validate_non_negative_index("doc", "N", N);
        doc = std::vector<int>(N,int(0));
        vals_i__ = context__.vals_i("doc");
        pos__ = 0;
        size_t doc_limit_0__ = N;
        for (size_t i_0__ = 0; i_0__ < doc_limit_0__; ++i_0__) {
            doc[i_0__] = vals_i__[pos__++];
        }
        validate_non_negative_index("alpha", "K", K);
        alpha = vector_d(static_cast<Eigen::VectorXd::Index>(K));
        context__.validate_dims("data initialization", "alpha", "vector_d", context__.to_vec(K));
        vals_r__ = context__.vals_r("alpha");
        pos__ = 0;
        size_t alpha_i_vec_lim__ = K;
        for (size_t i_vec__ = 0; i_vec__ < alpha_i_vec_lim__; ++i_vec__) {
            alpha[i_vec__] = vals_r__[pos__++];
        }
        validate_non_negative_index("beta", "V", V);
        beta = vector_d(static_cast<Eigen::VectorXd::Index>(V));
        context__.validate_dims("data initialization", "beta", "vector_d", context__.to_vec(V));
        vals_r__ = context__.vals_r("beta");
        pos__ = 0;
        size_t beta_i_vec_lim__ = V;
        for (size_t i_vec__ = 0; i_vec__ < beta_i_vec_lim__; ++i_vec__) {
            beta[i_vec__] = vals_r__[pos__++];
        }

        // validate, data variables
        check_greater_or_equal(function__,"K",K,2);
        check_greater_or_equal(function__,"V",V,2);
        check_greater_or_equal(function__,"M",M,1);
        check_greater_or_equal(function__,"N",N,1);
        for (int k0__ = 0; k0__ < N; ++k0__) {
            check_greater_or_equal(function__,"w[k0__]",w[k0__],1);
            check_less_or_equal(function__,"w[k0__]",w[k0__],V);
        }
        for (int k0__ = 0; k0__ < N; ++k0__) {
            check_greater_or_equal(function__,"doc[k0__]",doc[k0__],1);
            check_less_or_equal(function__,"doc[k0__]",doc[k0__],M);
        }
        check_greater_or_equal(function__,"alpha",alpha,0);
        check_greater_or_equal(function__,"beta",beta,0);
        // initialize data variables

        try {
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e,current_statement_begin__);
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }

        // validate transformed data

        // set parameter ranges
        num_params_r__ = 0U;
        param_ranges_i__.clear();
        num_params_r__ += (K - 1) * M;
        num_params_r__ += (V - 1) * K;
    }

    ~LDA_model() { }


    void transform_inits(const stan::io::var_context& context__,
                         std::vector<int>& params_i__,
                         std::vector<double>& params_r__,
                         std::ostream* pstream__) const {
        stan::io::writer<double> writer__(params_r__,params_i__);
        size_t pos__;
        (void) pos__; // dummy call to supress warning
        std::vector<double> vals_r__;
        std::vector<int> vals_i__;

        if (!(context__.contains_r("theta")))
            throw std::runtime_error("variable theta missing");
        vals_r__ = context__.vals_r("theta");
        pos__ = 0U;
        context__.validate_dims("initialization", "theta", "vector_d", context__.to_vec(M,K));
        // generate_declaration theta
        std::vector<vector_d> theta(M,vector_d(static_cast<Eigen::VectorXd::Index>(K)));
        for (int j1__ = 0U; j1__ < K; ++j1__)
            for (int i0__ = 0U; i0__ < M; ++i0__)
                theta[i0__](j1__) = vals_r__[pos__++];
        for (int i0__ = 0U; i0__ < M; ++i0__)
            try {
            writer__.simplex_unconstrain(theta[i0__]);
        } catch (const std::exception& e) { 
            throw std::runtime_error(std::string("Error transforming variable theta: ") + e.what());
        }

        if (!(context__.contains_r("phi")))
            throw std::runtime_error("variable phi missing");
        vals_r__ = context__.vals_r("phi");
        pos__ = 0U;
        context__.validate_dims("initialization", "phi", "vector_d", context__.to_vec(K,V));
        // generate_declaration phi
        std::vector<vector_d> phi(K,vector_d(static_cast<Eigen::VectorXd::Index>(V)));
        for (int j1__ = 0U; j1__ < V; ++j1__)
            for (int i0__ = 0U; i0__ < K; ++i0__)
                phi[i0__](j1__) = vals_r__[pos__++];
        for (int i0__ = 0U; i0__ < K; ++i0__)
            try {
            writer__.simplex_unconstrain(phi[i0__]);
        } catch (const std::exception& e) { 
            throw std::runtime_error(std::string("Error transforming variable phi: ") + e.what());
        }

        params_r__ = writer__.data_r();
        params_i__ = writer__.data_i();
    }

    void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                         std::ostream* pstream__) const {
      std::vector<double> params_r_vec;
      std::vector<int> params_i_vec;
      transform_inits(context, params_i_vec, params_r_vec, pstream__);
      params_r.resize(params_r_vec.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r(i) = params_r_vec[i];
    }


    template <bool propto__, bool jacobian__, typename T__>
    T__ log_prob(vector<T__>& params_r__,
                 vector<int>& params_i__,
                 std::ostream* pstream__ = 0) const {

        T__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // suppress unused var warning

        T__ lp__(0.0);
        stan::math::accumulator<T__> lp_accum__;

        // model parameters
        stan::io::reader<T__> in__(params_r__,params_i__);

        vector<Eigen::Matrix<T__,Eigen::Dynamic,1> > theta;
        size_t dim_theta_0__ = M;
        theta.reserve(dim_theta_0__);
        for (size_t k_0__ = 0; k_0__ < dim_theta_0__; ++k_0__) {
            if (jacobian__)
                theta.push_back(in__.simplex_constrain(K,lp__));
            else
                theta.push_back(in__.simplex_constrain(K));
        }

        vector<Eigen::Matrix<T__,Eigen::Dynamic,1> > phi;
        size_t dim_phi_0__ = K;
        phi.reserve(dim_phi_0__);
        for (size_t k_0__ = 0; k_0__ < dim_phi_0__; ++k_0__) {
            if (jacobian__)
                phi.push_back(in__.simplex_constrain(V,lp__));
            else
                phi.push_back(in__.simplex_constrain(V));
        }


        // transformed parameters


        try {
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e,current_statement_begin__);
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }

        // validate transformed parameters

        const char* function__ = "validate transformed params";
        (void) function__;  // dummy to suppress unused var warning

        // model body
        try {

            current_statement_begin__ = 16;
            for (int m = 1; m <= M; ++m) {
                current_statement_begin__ = 17;
                lp_accum__.add(dirichlet_log<propto__>(get_base1(theta,m,"theta",1), alpha));
            }
            current_statement_begin__ = 18;
            for (int k = 1; k <= K; ++k) {
                current_statement_begin__ = 19;
                lp_accum__.add(dirichlet_log<propto__>(get_base1(phi,k,"phi",1), beta));
            }
            current_statement_begin__ = 20;
            for (int n = 1; n <= N; ++n) {
                {
                    vector<T__> gamma(K);
                    stan::math::initialize(gamma, DUMMY_VAR__);
                    stan::math::fill(gamma,DUMMY_VAR__);


                    current_statement_begin__ = 22;
                    for (int k = 1; k <= K; ++k) {
                        current_statement_begin__ = 23;
                        stan::math::assign(get_base1_lhs(gamma,k,"gamma",1), (log(get_base1(get_base1(theta,get_base1(doc,n,"doc",1),"theta",1),k,"theta",2)) + log(get_base1(get_base1(phi,k,"phi",1),get_base1(w,n,"w",1),"phi",2))));
                    }
                    current_statement_begin__ = 24;
                    lp_accum__.add(log_sum_exp(gamma));
                }
            }
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e,current_statement_begin__);
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }

        lp_accum__.add(lp__);
        return lp_accum__.sum();

    } // log_prob()

    template <bool propto, bool jacobian, typename T_>
    T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
               std::ostream* pstream = 0) const {
      std::vector<T_> vec_params_r;
      vec_params_r.reserve(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        vec_params_r.push_back(params_r(i));
      std::vector<int> vec_params_i;
      return log_prob<propto,jacobian,T_>(vec_params_r, vec_params_i, pstream);
    }


    void get_param_names(std::vector<std::string>& names__) const {
        names__.resize(0);
        names__.push_back("theta");
        names__.push_back("phi");
    }


    void get_dims(std::vector<std::vector<size_t> >& dimss__) const {
        dimss__.resize(0);
        std::vector<size_t> dims__;
        dims__.resize(0);
        dims__.push_back(M);
        dims__.push_back(K);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dims__.push_back(K);
        dims__.push_back(V);
        dimss__.push_back(dims__);
    }

    template <typename RNG>
    void write_array(RNG& base_rng__,
                     std::vector<double>& params_r__,
                     std::vector<int>& params_i__,
                     std::vector<double>& vars__,
                     bool include_tparams__ = true,
                     bool include_gqs__ = true,
                     std::ostream* pstream__ = 0) const {
        vars__.resize(0);
        stan::io::reader<double> in__(params_r__,params_i__);
        static const char* function__ = "LDA_model_namespace::write_array";
        (void) function__; // dummy call to supress warning
        // read-transform, write parameters
        vector<vector_d> theta;
        size_t dim_theta_0__ = M;
        for (size_t k_0__ = 0; k_0__ < dim_theta_0__; ++k_0__) {
            theta.push_back(in__.simplex_constrain(K));
        }
        vector<vector_d> phi;
        size_t dim_phi_0__ = K;
        for (size_t k_0__ = 0; k_0__ < dim_phi_0__; ++k_0__) {
            phi.push_back(in__.simplex_constrain(V));
        }
        for (int k_1__ = 0; k_1__ < K; ++k_1__) {
            for (int k_0__ = 0; k_0__ < M; ++k_0__) {
                vars__.push_back(theta[k_0__][k_1__]);
            }
        }
        for (int k_1__ = 0; k_1__ < V; ++k_1__) {
            for (int k_0__ = 0; k_0__ < K; ++k_0__) {
                vars__.push_back(phi[k_0__][k_1__]);
            }
        }

        if (!include_tparams__) return;
        // declare and define transformed parameters
        double lp__ = 0.0;
        (void) lp__; // dummy call to supress warning
        stan::math::accumulator<double> lp_accum__;

        double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // suppress unused var warning



        try {
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e,current_statement_begin__);
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }

        // validate transformed parameters

        // write transformed parameters

        if (!include_gqs__) return;
        // declare and define generated quantities


        try {
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e,current_statement_begin__);
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }

        // validate generated quantities

        // write generated quantities
    }

    template <typename RNG>
    void write_array(RNG& base_rng,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                     bool include_tparams = true,
                     bool include_gqs = true,
                     std::ostream* pstream = 0) const {
      std::vector<double> params_r_vec(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r_vec[i] = params_r(i);
      std::vector<double> vars_vec;
      std::vector<int> params_i_vec;
      write_array(base_rng,params_r_vec,params_i_vec,vars_vec,include_tparams,include_gqs,pstream);
      vars.resize(vars_vec.size());
      for (int i = 0; i < vars.size(); ++i)
        vars(i) = vars_vec[i];
    }

    static std::string model_name() {
        return "LDA_model";
    }


    void constrained_param_names(std::vector<std::string>& param_names__,
                                 bool include_tparams__ = true,
                                 bool include_gqs__ = true) const {
        std::stringstream param_name_stream__;
        for (int k_1__ = 1; k_1__ <= K; ++k_1__) {
            for (int k_0__ = 1; k_0__ <= M; ++k_0__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "theta" << '.' << k_0__ << '.' << k_1__;
                param_names__.push_back(param_name_stream__.str());
            }
        }
        for (int k_1__ = 1; k_1__ <= V; ++k_1__) {
            for (int k_0__ = 1; k_0__ <= K; ++k_0__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "phi" << '.' << k_0__ << '.' << k_1__;
                param_names__.push_back(param_name_stream__.str());
            }
        }

        if (!include_gqs__ && !include_tparams__) return;

        if (!include_gqs__) return;
    }


    void unconstrained_param_names(std::vector<std::string>& param_names__,
                                   bool include_tparams__ = true,
                                   bool include_gqs__ = true) const {
        std::stringstream param_name_stream__;
        for (int k_1__ = 1; k_1__ <= (K - 1); ++k_1__) {
            for (int k_0__ = 1; k_0__ <= M; ++k_0__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "theta" << '.' << k_0__ << '.' << k_1__;
                param_names__.push_back(param_name_stream__.str());
            }
        }
        for (int k_1__ = 1; k_1__ <= (V - 1); ++k_1__) {
            for (int k_0__ = 1; k_0__ <= K; ++k_0__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "phi" << '.' << k_0__ << '.' << k_1__;
                param_names__.push_back(param_name_stream__.str());
            }
        }

        if (!include_gqs__ && !include_tparams__) return;

        if (!include_gqs__) return;
    }

}; // model

} // namespace

typedef LDA_model_namespace::LDA_model stan_model;

