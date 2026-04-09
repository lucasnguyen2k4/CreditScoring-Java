/**
 * RBAC permissions — port of original utils/permissions.py
 */

export const PAGE_ACCESS = {
  '/dashboard':           ['ADMIN', 'MODEL_BUILDER', 'VALIDATOR', 'SCORER'],
  '/upload':              ['ADMIN', 'MODEL_BUILDER', 'VALIDATOR'],
  '/feature-engineering': ['ADMIN', 'MODEL_BUILDER', 'VALIDATOR'],
  '/training':            ['ADMIN', 'MODEL_BUILDER', 'VALIDATOR'],
  '/model-approval':      ['ADMIN', 'MODEL_BUILDER', 'VALIDATOR'],
  '/shap':                ['ADMIN', 'MODEL_BUILDER', 'VALIDATOR'],
  '/prediction':          ['ADMIN', 'MODEL_BUILDER', 'SCORER'],
  '/admin':               ['ADMIN'],
};

export const VIEW_ONLY_PAGES = {
  VALIDATOR: ['/upload', '/feature-engineering', '/training', '/shap'],
};

export const canAccess = (role, path) => {
  const allowed = PAGE_ACCESS[path];
  return allowed ? allowed.includes(role) : false;
};

export const isViewOnly = (role, path) => {
  const pages = VIEW_ONLY_PAGES[role];
  return pages ? pages.includes(path) : false;
};

export const getNavItems = (role) => {
  const items = [
    { path: '/dashboard', label: 'Dashboard', icon: 'LayoutDashboard' },
    { path: '/upload', label: 'Data Upload & EDA', icon: 'Upload' },
    { path: '/feature-engineering', label: 'Feature Engineering', icon: 'Settings' },
    { path: '/training', label: 'Model Training', icon: 'Brain' },
    { path: '/model-approval', label: 'Model Approval', icon: 'ShieldCheck' },
    { path: '/shap', label: 'SHAP Explanation', icon: 'BarChart3' },
    { path: '/prediction', label: 'Prediction', icon: 'Target' },
    { path: '/admin', label: 'Admin Settings', icon: 'Shield' },
  ];
  return items.filter((item) => canAccess(role, item.path));
};
