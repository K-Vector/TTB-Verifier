import { Route, Switch } from "wouter";
import VerificationPage from "./pages/VerificationPage";
import NotFound from "./pages/NotFound";

function Router() {
  return (
    <Switch>
      <Route path="/" component={VerificationPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return <Router />;
}

export default App;
