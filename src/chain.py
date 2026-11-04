"""Local Ethereum (py-evm via eth-tester) deployment of FederatedAuditRegistry."""
import hashlib
import time
from pathlib import Path

import solcx
from web3 import EthereumTesterProvider, Web3
from eth_tester.exceptions import TransactionFailed
from web3.exceptions import ContractLogicError

SOLC = "0.8.24"
CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "FederatedAuditRegistry.sol"


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def compile_contract():
    solcx.set_solc_version(SOLC)
    out = solcx.compile_files([str(CONTRACT)], output_values=["abi", "bin"],
                              optimize=True, optimize_runs=200)
    (_, c), = [(k, v) for k, v in out.items() if k.endswith(":FederatedAuditRegistry")]
    return c["abi"], c["bin"]


class AuditChain:
    """Thin wrapper that records gas and wall-clock latency of every transaction."""

    def __init__(self, n_sites: int, extra_accounts: int = 2):
        self.w3 = Web3(EthereumTesterProvider())
        acc = self.w3.eth.accounts
        need = 2 + n_sites + extra_accounts
        if len(acc) < need:
            # eth-tester ships 10 funded accounts; add more for scalability runs
            prov = self.w3.provider.ethereum_tester
            for i in range(need - len(acc)):
                pk = "0x" + hashlib.sha256(f"node-{i}".encode()).hexdigest()
                addr = prov.add_account(pk)
                prov.send_transaction({"from": acc[0], "to": addr, "value": 10**21,
                                       "gas": 21000, "gas_price": 10**9})
            acc = self.w3.eth.accounts
        self.governor, self.aggregator = acc[0], acc[1]
        self.nodes = list(acc[2:2 + n_sites])
        self.outsiders = list(acc[2 + n_sites:need])
        self.log = []
        abi, bytecode = compile_contract()
        t = time.perf_counter()
        tx = self.w3.eth.contract(abi=abi, bytecode=bytecode).constructor(self.aggregator).transact({"from": self.governor})
        rc = self.w3.eth.wait_for_transaction_receipt(tx)
        self.log.append(("deploy", rc.gasUsed, time.perf_counter() - t, True))
        self.c = self.w3.eth.contract(address=rc.contractAddress, abi=abi)
        self.bytecode_size = len(bytes.fromhex(bytecode)) if not bytecode.startswith("0x") else len(bytes.fromhex(bytecode[2:]))

    def _tx(self, op, fn, sender):
        t = time.perf_counter()
        try:
            tx = fn.transact({"from": sender})
            rc = self.w3.eth.wait_for_transaction_receipt(tx)
            ok = rc.status == 1
            self.log.append((op, rc.gasUsed, time.perf_counter() - t, ok))
            return rc
        except (ContractLogicError, TransactionFailed) as e:
            self.log.append((op, 0, time.perf_counter() - t, False))
            raise PermissionError(str(e)) from None

    # --- protocol operations
    def register(self, node, site_id: str):
        return self._tx("registerNode", self.c.functions.registerNode(node, site_id.encode().ljust(32, b"\0")), self.governor)

    def open_round(self, r, threshold_auroc):
        return self._tx("openRound", self.c.functions.openRound(r, int(round(threshold_auroc * 1e4))), self.aggregator)

    def submit(self, node, r, model_bytes: bytes, meta_bytes: bytes, n_train: int):
        return self._tx("submitModel", self.c.functions.submitModel(r, sha256(model_bytes), sha256(meta_bytes), n_train), node)

    def attest(self, validator, r, owner, evaluated_bytes: bytes, auroc: float, eval_record: bytes):
        return self._tx("attest", self.c.functions.attest(r, owner, sha256(evaluated_bytes),
                                                          int(round(auroc * 1e4)), sha256(eval_record)), validator)

    def finalize(self, r):
        return self._tx("finalizeRound", self.c.functions.finalizeRound(r), self.aggregator)

    def commit_global(self, r, global_bytes: bytes, weights_bytes: bytes):
        return self._tx("commitGlobal", self.c.functions.commitGlobal(r, sha256(global_bytes), sha256(weights_bytes)), self.aggregator)

    # --- views
    def verify(self, r, node, model_bytes: bytes) -> bool:
        return self.c.functions.verifyModel(r, node, sha256(model_bytes)).call()

    def submission(self, r, node):
        h, meta, n, acc, med, k = self.c.functions.getSubmission(r, node).call()
        return {"accepted": acc, "median_auroc": med / 1e4, "n_attestations": k, "n_train": n}

    def decisions(self, r):
        ev = self.c.events.ModelDecision.get_logs(from_block=0, argument_filters={"round": r})
        return {e.args.node: (e.args.accepted, e.args.medianScore / 1e4) for e in ev}
