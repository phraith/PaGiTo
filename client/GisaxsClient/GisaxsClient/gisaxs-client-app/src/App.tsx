import Simulation from "./components/Simulation/Simulation";

import { BrowserRouter, Routes, Route } from "react-router-dom";
import Fitting from "./components/Fitting/Fitting";
import React from "react";
import Jobs from "./components/Jobs/Jobs";

const App = () => {
  return (
    <React.Fragment>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Simulation />} />
          <Route path="simulation" element={<Simulation />} />
          <Route path="fitting" element={<Fitting />} />
          <Route path="jobs" element={<Jobs />} />
        </Routes>
      </BrowserRouter>
    </React.Fragment>
  );
};

export default App;
