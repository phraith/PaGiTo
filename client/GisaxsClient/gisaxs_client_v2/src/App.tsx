import Simulation from "./components/Simulation/Simulation";

import { BrowserRouter, Routes, Route } from "react-router-dom";
import Fitting from "./components/Fitting/Fitting";
import React from "react";

const App = () => {
  return (
    <React.Fragment>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Simulation />} />
          <Route path="simulation" element={<Simulation />} />
          <Route path="fitting" element={<Fitting />} />
        </Routes>
      </BrowserRouter>
    </React.Fragment>
  );
};

export default App;
