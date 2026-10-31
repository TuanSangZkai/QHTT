import io
import os
import sys
import json
import zlib
import base64
from fractions import Fraction
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Bảng ký tự Unicode subscript cho chỉ số dưới
_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUBSCRIPT_LETTERS = {"i": "ᵢ", "j": "ⱼ", "B": "ᴮ", "n": "ₙ", "m": "ₘ", "k": "ₖ"}


def _sub(text):
    """Chuyển các chữ số và chữ cái phổ biến sang dạng subscript Unicode."""
    result = []
    for ch in text:
        if ch in _SUBSCRIPT_LETTERS:
            result.append(_SUBSCRIPT_LETTERS[ch])
        else:
            result.append(ch.translate(_SUBSCRIPT_DIGITS))
    return "".join(result)


# =====================================================================
# PHẦN 1: THUẬT TOÁN QUY HOẠCH TUYẾN TÍNH (SIMPLEX & BIG-M)
# =====================================================================
class LinearProgrammingSolver:

    @staticmethod
    def _fraction(value):
        if abs(value) < 1e-9:
            return "0"
        return str(Fraction(float(value)).limit_denominator(1000000))

    def _big_m(self, coefficient, constant):
        if abs(coefficient) < 1e-9:
            return self._fraction(constant)
        magnitude = self._fraction(abs(coefficient))
        result = ("-" if coefficient < 0 else "") + ("" if magnitude == "1" else magnitude + "*") + "M"
        if abs(constant) >= 1e-9:
            result += (" + " if constant > 0 else " - ") + self._fraction(abs(constant))
        return result

    def __init__(self, c, A, b, signs, mode="max"):
        self.c_orig = np.array(c, dtype=float)
        self.A_orig = np.array(A, dtype=float)
        self.b_orig = np.array(b, dtype=float)
        self.signs = [s.replace("≤", "<=").replace("≥", ">=") for s in signs]
        self.mode = mode.lower()
        self.M = 1e6

    def print_dual(self):
        """Lập đối ngẫu trực tiếp từ đề gốc, với các biến gốc xⱼ ≥ 0."""
        m, n = self.A_orig.shape
        names = [f"y{_sub(str(i+1))}" for i in range(m)]

        def expression(coefficients, variables=None):
            variables = names if variables is None else variables
            terms = []
            for value, name in zip(coefficients, variables):
                if value == 0:
                    continue
                term = ("" if abs(value) == 1 else self._fraction(abs(value)) + "·") + name
                if terms:
                    term = (" - " if value < 0 else " + ") + term
                elif value < 0:
                    term = "-" + term
                terms.append(term)
            return "".join(terms) or "0"

        dual_mode = "MIN" if self.mode == "max" else "MAX"
        relation = "≥" if self.mode == "max" else "≤"
        sign_rules = ({"<=": "≥ 0", ">=": "≤ 0", "=": "tự do"}
                      if self.mode == "max" else
                      {"<=": "≤ 0", ">=": "≥ 0", "=": "tự do"})
        def print_system(rows):
            middle = len(rows) // 2
            for index, row in enumerate(rows):
                brace = "⎧" if index == 0 else "⎩" if index == len(rows) - 1 else "⎨" if index == middle else "⎪"
                print(f"    {brace} {row}")

        original_names = [f"x{_sub(str(j+1))}" for j in range(n)]
        relations = {"<=": "≤", ">=": "≥", "=": "="}
        print("\n[BÀI TOÁN GỐC]:")
        print(f"  Z = {expression(self.c_orig, original_names)} → {self.mode.upper()}")
        original_rows = [
            f"{expression(self.A_orig[i], original_names)} {relations[self.signs[i]]} {self._fraction(self.b_orig[i])}"
            for i in range(m)
        ]
        original_rows.append(", ".join(original_names) + " ≥ 0")
        print_system(original_rows)

        print("\n[BÀI TOÁN ĐỐI NGẪU]:")
        print(f"  W = {expression(self.b_orig)} → {dual_mode}")
        dual_rows = [
            f"{expression(self.A_orig[:, j])} {relation} {self._fraction(self.c_orig[j])}"
            for j in range(n)
        ]
        dual_rows.append("; ".join(f"{names[i]} {sign_rules[self.signs[i]]}" for i in range(m)))
        print_system(dual_rows)

    def solve(self):
        print("===================================================================")
        print("                 GIẢI BÀI TOÁN QUY HOẠCH TUYẾN TÍNH ")
        print("===================================================================")

        m, n = self.A_orig.shape

        self.print_dual()

        for i in range(m):
            if self.b_orig[i] < 0:
                print(f"  + Ràng buộc ({i+1}): nhân hai vế với -1 để vế phải không âm; đảo chiều bất đẳng thức.")
                self.A_orig[i] *= -1
                self.b_orig[i] *= -1
                if self.signs[i] == "<=":
                    self.signs[i] = ">="
                elif self.signs[i] == ">=":
                    self.signs[i] = "<="

        def expression(coefficients, names):
            terms = []
            for coefficient, name in zip(coefficients, names):
                if coefficient == 0:
                    continue
                magnitude = "" if abs(coefficient) == 1 else self._fraction(abs(coefficient)) + "·"
                term = magnitude + name
                if not terms:
                    terms.append(("-" if coefficient < 0 else "") + term)
                else:
                    terms.append((" - " if coefficient < 0 else " + ") + term)
            return "".join(terms) or "0"

        print("\n[BƯỚC CHUẨN HÓA: CHUYỂN CÁC RÀNG BUỘC VỀ DẤU BẰNG]:")
        print("  Dấu ≤: cộng biến phụ không âm; dấu ≥: trừ biến phụ không âm; dấu =: giữ nguyên.")
        print(f"  Biến phụ được đánh số tiếp sau {n} biến ban đầu; hệ số mục tiêu bằng 0.")
        auxiliary_names = {}
        for i in range(m):
            if self.signs[i] in ("<=", ">="):
                auxiliary_names[i] = f"x{_sub(str(n + len(auxiliary_names) + 1))}"
        original_names = [f"x{_sub(str(j+1))}" for j in range(n)]
        for i in range(m):
            lhs = expression(self.A_orig[i], original_names)
            if self.signs[i] == "<=":
                lhs += f" + {auxiliary_names[i]}"
            elif self.signs[i] == ">=":
                lhs += f" - {auxiliary_names[i]}"
            print(f"    ({i+1}) {lhs} = {self._fraction(self.b_orig[i])}")
        print("  Các biến xⱼ đều không âm. Hệ dấu bằng này tương đương các ràng buộc ban đầu.")

        var_names = [f"x{_sub(str(i+1))}" for i in range(n)]
        c_ext = list(self.c_orig if self.mode == "max" else -self.c_orig)

        basis = []
        col_idx = n

        for i in range(m):
            if self.signs[i] == "<=":
                var_names.append(auxiliary_names[i])
                c_ext.append(0.0)
                basis.append(col_idx)
                col_idx += 1
            elif self.signs[i] == ">=":
                var_names.append(auxiliary_names[i])
                c_ext.append(0.0)
                col_idx += 1
                var_names.append(f"a{_sub(str(i+1))}")
                c_ext.append(-self.M)
                basis.append(col_idx)
                col_idx += 1
            elif self.signs[i] == "=":
                var_names.append(f"a{_sub(str(i+1))}")
                c_ext.append(-self.M)
                basis.append(col_idx)
                col_idx += 1

        total_vars = col_idx
        artificial = np.array([name.startswith("a") for name in var_names])
        column_names = [f"A{_sub(str(j+1))}" for j in range(total_vars)]
        print("\n[KÝ HIỆU CÁC CỘT]: " + ", ".join(
            f"{column_names[j]} = {var_names[j]}" for j in range(total_vars)))
        A_ext = np.zeros((m, total_vars))
        A_ext[:, :n] = self.A_orig

        curr_col = n
        for i in range(m):
            if self.signs[i] == "<=":
                A_ext[i, curr_col] = 1.0
                curr_col += 1
            elif self.signs[i] == ">=":
                A_ext[i, curr_col] = -1.0
                A_ext[i, curr_col + 1] = 1.0
                curr_col += 2
            elif self.signs[i] == "=":
                A_ext[i, curr_col] = 1.0
                curr_col += 1

        c_ext = np.array(c_ext, dtype=float)
        b = self.b_orig.copy()

        print("\n[HỆ PHƯƠNG TRÌNH DÙNG LẬP BẢNG ĐƠN HÌNH]:")
        if np.any(artificial):
            print("  Thêm biến giả aᵢ ≥ 0 vào các hàng ≥ hoặc = để tạo cơ sở ban đầu.")
            print("  Biến giả phải bằng 0 ở nghiệm hợp lệ; hệ có biến giả chỉ là bài toán hỗ trợ.")
        for i in range(m):
            print(f"    ({i+1}) {expression(A_ext[i], var_names)} = {self._fraction(b[i])}")
        objective = expression(np.where(artificial, 0.0, c_ext), var_names)
        for j in range(total_vars):
            if artificial[j]:
                objective += f" - M·{var_names[j]}"
        objective_name = "Z" if self.mode == "max" else "Z' = -Z"
        print(f"  Hàm mục tiêu dùng tính bảng: {objective_name} = {objective} --> MAX")
        print("  Tất cả các biến trong bảng đều không âm.")

        step = 1
        while True:
            print(f"\n--- BƯỚC {step}: BẢNG ĐƠN HÌNH ---")
            c_B = c_ext[basis]
            delta = np.zeros(total_vars)
            for j in range(total_vars):
                delta[j] = np.dot(c_B, A_ext[:, j]) - c_ext[j]

            pivot_cell = None
            if not np.all(delta >= -1e-7):
                pivot_col = int(np.argmin(delta))
                pivot_ratios = [b[i] / A_ext[i, pivot_col]
                                if A_ext[i, pivot_col] > 1e-7 else np.inf
                                for i in range(m)]
                if any(np.isfinite(ratio) for ratio in pivot_ratios):
                    pivot_cell = (int(np.argmin(pivot_ratios)), pivot_col)
            headers = ["Bước", "Hệ số", "Cơ sở Aⱼ", "Phương án"] + column_names
            rows = []
            for i in range(m):
                cb_str = "-M" if artificial[basis[i]] else self._fraction(c_B[i])
                cells = [(f"⟦{self._fraction(value)}⟧" if pivot_cell == (i, j)
                          else self._fraction(value)) for j, value in enumerate(A_ext[i])]
                rows.append([str(step) if i == m // 2 else "", cb_str, column_names[basis[i]], self._fraction(b[i])] + cells)
            finite_costs = np.where(artificial, 0.0, c_ext)
            m_costs = -artificial.astype(float)
            delta_values = []
            for j in range(total_vars):
                m_coef = np.dot(m_costs[basis], A_ext[:, j]) - m_costs[j]
                constant = np.dot(finite_costs[basis], A_ext[:, j]) - finite_costs[j]
                delta_values.append(self._big_m(m_coef, constant))
            delta_row = ["", "", "Δⱼ", ""] + delta_values
            widths = [max(len(row[j]) for row in [headers] + rows + [delta_row])
                      for j in range(len(headers))]

            def format_row(row):
                return " | ".join(value.ljust(width) for value, width in zip(row, widths))

            header = format_row(headers)
            print(header)
            print("-" * len(header))
            for row in rows:
                print(format_row(row))
            print("-" * len(header))
            print(format_row(delta_row))

            if np.all(delta >= -1e-7):
                print("\n=> Mọi hệ số ước lượng Δⱼ ≥ 0. Đã đạt phương án tối ưu!")
                break

            in_col = int(np.argmin(delta))
            print(f"\n-> Biến đưa vào cơ sở: {column_names[in_col]} (có Δⱼ âm nhỏ nhất)")

            ratios = []
            for i in range(m):
                if A_ext[i, in_col] > 1e-7:
                    ratios.append(b[i] / A_ext[i, in_col])
                else:
                    ratios.append(np.inf)

            if all(r == np.inf for r in ratios):
                print("\n[KẾT LUẬN]: Mọi hệ số trên cột xoay <= 0. Bài toán KHÔNG GIỚI NỘI (Unbounded)!")
                return None, None

            out_row = int(np.argmin(ratios))
            print(
                f"-> Biến đưa ra khỏi cơ sở: {column_names[basis[out_row]]} (Tỷ số θ"
                f" nhỏ nhất = {self._fraction(ratios[out_row])} tại hàng {out_row+1})"
            )
            print(
                f"-> Phần tử xoay (Pivot) = {self._fraction(A_ext[out_row, in_col])} tại (Hàng"
                f" {out_row+1}, Cột {column_names[in_col]})"
            )

            pivot = A_ext[out_row, in_col]
            A_ext[out_row, :] /= pivot
            b[out_row] /= pivot

            for i in range(m):
                if i != out_row:
                    factor = A_ext[i, in_col]
                    A_ext[i, :] -= factor * A_ext[out_row, :]
                    b[i] -= factor * b[out_row]

            basis[out_row] = in_col
            step += 1

        for i in range(m):
            if var_names[basis[i]].startswith("a") and b[i] > 1e-5:
                print("\n[KẾT LUẬN]: Tồn tại biến giả dương trong cơ sở tối ưu. Bài toán VÔ NGHIỆM!")
                return None, None

        x_opt = np.zeros(n)
        for i in range(m):
            if basis[i] < n:
                x_opt[basis[i]] = b[i]

        z_opt = np.dot(self.c_orig, x_opt)
        print("\n************************************************************")
        print("KẾT QUẢ TỐI ƯU (QUY HOẠCH TUYẾN TÍNH):")
        for i in range(n):
            print(f"  + x{_sub(str(i+1))} = {self._fraction(x_opt[i])}")
        print(f"  => Giá trị hàm mục tiêu Z({self.mode.upper()}) = {self._fraction(z_opt)}")
        print("************************************************************")
        return x_opt, z_opt


# =====================================================================
# PHẦN 2: THUẬT TOÁN BÀI TOÁN VẬN TẢI
# =====================================================================
class TransportationSolver:

    def __init__(self, supply, demand, costs):
        self.supply_orig = np.array(supply, dtype=float)
        self.demand_orig = np.array(demand, dtype=float)
        self.costs_orig = np.array(costs, dtype=float)

    def _balance_problem(self):
        total_s = np.sum(self.supply_orig)
        total_d = np.sum(self.demand_orig)
        self.supply = self.supply_orig.copy()
        self.demand = self.demand_orig.copy()
        self.costs = self.costs_orig.copy()

        print("\n--- BƯỚC 1: KIỂM TRA CÂN BẰNG THU - PHÁT ---")
        print(f"  + Tổng lượng phát = {total_s:g}")
        print(f"  + Tổng lượng thu  = {total_d:g}")

        if abs(total_s - total_d) < 1e-7:
            print("=> Bài toán Vận tải ĐÃ CÂN BẰNG (Thu = Phát).")
        elif total_s > total_d:
            diff = total_s - total_d
            print(f"=> Tổng Phát > Tổng Thu. Thêm 1 TRẠM THU GIẢ với nhu cầu = {diff:g} và cước phí = 0.")
            self.demand = np.append(self.demand, diff)
            zero_col = np.zeros((self.costs.shape[0], 1))
            self.costs = np.hstack((self.costs, zero_col))
        else:
            diff = total_d - total_s
            print(f"=> Tổng Thu > Tổng Phát. Thêm 1 TRẠM PHÁT GIẢ với khả năng = {diff:g} và cước phí = 0.")
            self.supply = np.append(self.supply, diff)
            zero_row = np.zeros((1, self.costs.shape[1]))
            self.costs = np.vstack((self.costs, zero_row))

    def _print_plan(self, X, basis):
        m, n = self.costs.shape

        # ---- 1. Tách mỗi ô thành (phần phân bổ, phần chi phí) ----
        left = [[(f"{X[i, j]:.0f}" if (i, j) in basis else ".") for j in range(n)]
                for i in range(m)]
        cost = [[f"{self.costs[i, j]:.0f}" for j in range(n)] for i in range(m)]

        # Độ rộng từng phần theo từng cột để các "[ ]" thẳng hàng
        left_w = [max(len(left[i][j]) for i in range(m)) for j in range(n)]
        cost_w = [max(len(cost[i][j]) for i in range(m)) for j in range(n)]

        # Ghép lại: mọi ô trong cùng một cột có độ dài bằng nhau
        rows = [[f"{left[i][j]:>{left_w[j]}} [{cost[i][j]:>{cost_w[j]}}]" for j in range(n)]
                for i in range(m)]

        head = [f"B{_sub(str(j+1))}" for j in range(n)]
        row_labels = [f"A{_sub(str(i+1))}" for i in range(m)]
        supply_col = [f"{s:.0f}" for s in self.supply]
        demand_row = [f"{d:.0f}" for d in self.demand]
        total = f"{np.sum(self.supply):.0f}"

        # ---- 2. Độ rộng mỗi cột theo nội dung dài nhất ----
        label_w = max(len("Thu bⱼ"), *(len(s) for s in row_labels))
        col_w = [
            max(len(head[j]), len(demand_row[j]), len(rows[0][j]))
            for j in range(n)
        ]
        last_w = max(len("Phát (aᵢ)"), len(total), *(len(s) for s in supply_col))

        # ---- 3. Hàm in một dòng dùng chung ----
        def fmt_line(label, cells, last):
            body = " | ".join(c.center(w) for c, w in zip(cells, col_w))
            return f" {label:<{label_w}} | {body} | {last:^{last_w}} "

        header = fmt_line("", head, "Phát (aᵢ)")
        sep = "-" * len(header)

        # ---- 4. In bảng ----
        print(header)
        print(sep)
        for i in range(m):
            print(fmt_line(row_labels[i], rows[i], supply_col[i]))
        print(sep)
        print(fmt_line("Thu bⱼ", demand_row, total))

    def north_west_corner(self):
        print("\n--- BƯỚC 2: TÌM PHƯƠNG ÁN BAN ĐẦU BẰNG GÓC TÂY BẮC ---")
        m, n = self.costs.shape
        s, d = self.supply.copy(), self.demand.copy()
        X = np.zeros((m, n))
        basis = set()
        i, j = 0, 0
        while i < m and j < n:
            qty = min(s[i], d[j])
            X[i, j] = qty
            basis.add((i, j))
            s[i] -= qty
            d[j] -= qty
            print(f"  + Phân phối vào ô (A{_sub(str(i+1))}, B{_sub(str(j+1))}) lượng hàng x = {qty:g}")
            if abs(s[i]) < 1e-7 and abs(d[j]) < 1e-7:
                if j + 1 < n:
                    j += 1
                else:
                    i += 1
            elif abs(s[i]) < 1e-7:
                i += 1
            else:
                j += 1
        return X, basis

    def minimum_cost_method(self):
        print("\n--- BƯỚC 2: TÌM PHƯƠNG ÁN BAN ĐẦU BẰNG CƯỚC PHÍ NHỎ NHẤT ---")
        m, n = self.costs.shape
        s, d = self.supply.copy(), self.demand.copy()
        X = np.zeros((m, n))
        basis = set()
        active_rows, active_cols = set(range(m)), set(range(n))

        while active_rows and active_cols:
            min_c = np.inf
            best_cell = None
            for i in active_rows:
                for j in active_cols:
                    if self.costs[i, j] < min_c:
                        min_c = self.costs[i, j]
                        best_cell = (i, j)
            i, j = best_cell
            qty = min(s[i], d[j])
            X[i, j] = qty
            basis.add((i, j))
            s[i] -= qty
            d[j] -= qty
            print(f"  + Chọn ô cước phí nhỏ nhất (A{_sub(str(i+1))}, B{_sub(str(j+1))}) [c={min_c:g}], phân phối x = {qty:g}")
            if abs(s[i]) < 1e-7 and abs(d[j]) < 1e-7:
                if len(active_rows) > 1:
                    active_rows.remove(i)
                else:
                    active_cols.remove(j)
            elif abs(s[i]) < 1e-7:
                active_rows.remove(i)
            else:
                active_cols.remove(j)
        return X, basis

    def _ensure_non_degenerate(self, basis):
        m, n = self.costs.shape
        parent = list(range(m + n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
                return True
            return False

        clean_basis = set()
        for i, j in basis:
            if union(i, m + j):
                clean_basis.add((i, j))

        candidates = sorted([(self.costs[i, j], i, j) for i in range(m) for j in range(n)])
        for _, i, j in candidates:
            if len(clean_basis) == m + n - 1:
                break
            if (i, j) not in clean_basis:
                if union(i, m + j):
                    clean_basis.add((i, j))
                    print(f"  * [Suy biến]: Bổ sung ô chọn giả (A{_sub(str(i+1))}, B{_sub(str(j+1))}) với lượng hàng = 0 vào cơ sở.")
        return clean_basis

    def _find_cycle(self, basis, start_cell):
        nodes = list(basis) + [start_cell]

        def dfs(curr, path, look_row):
            if len(path) >= 4 and curr == start_cell:
                return path[:-1]
            r, c = curr
            if look_row:
                next_cells = [cell for cell in nodes if cell[0] == r and cell != curr]
            else:
                next_cells = [cell for cell in nodes if cell[1] == c and cell != curr]

            for nxt in next_cells:
                if nxt == start_cell and len(path) >= 4:
                    return path
                if nxt not in path:
                    res = dfs(nxt, path + [nxt], not look_row)
                    if res is not None:
                        return res
            return None

        return dfs(start_cell, [start_cell], True)

    def optimize_modi(self, X, basis):
        m, n = self.costs.shape
        step = 1
        while True:
            basis = self._ensure_non_degenerate(basis)
            print(f"\n{'='*70}")
            print(f"  BƯỚC TỐI ƯU THẾ VỊ - LẦN LẶP {step}")
            print(f"{'='*70}")

            print(f"\n[BẢNG PHƯƠNG ÁN HIỆN TẠI]:")
            self._print_plan(X, basis)
            total_cost = np.sum(X * self.costs)
            print(f"=> Tổng chi phí vận chuyển hiện tại: F = {total_cost:.0f}")

            u, v = [None] * m, [None] * n
            u[0] = 0.0
            changed = True
            while changed:
                changed = False
                for i, j in basis:
                    if u[i] is not None and v[j] is None:
                        v[j] = self.costs[i, j] - u[i]
                        changed = True
                    elif v[j] is not None and u[i] is None:
                        u[i] = self.costs[i, j] - v[j]
                        changed = True

            u_vals = [round(val, 2) if val is not None else 0 for val in u]
            v_vals = [round(val, 2) if val is not None else 0 for val in v]
            u_str = ", ".join([f"{x:g}" for x in u_vals])
            v_str = ", ".join([f"{x:g}" for x in v_vals])
            print(f"\n[TÍNH THẾ VỊ]: (Đặt u₁ = 0, dùng uᵢ + vⱼ = cᵢⱼ cho ô cơ sở)")
            print(f"  u = {{ {u_str} }}")
            print(f"  v = {{ {v_str} }}")

            delta_matrix = np.full((m, n), np.nan)
            max_delta = 0.0
            entering_cell = None
            for i in range(m):
                for j in range(n):
                    if (i, j) not in basis:
                        delta_ij = u_vals[i] + v_vals[j] - self.costs[i, j]
                        delta_matrix[i, j] = delta_ij
                        if delta_ij > max_delta + 1e-7:
                            max_delta = delta_ij
                            entering_cell = (i, j)

            print(f"\n[BẢNG HỆ SỐ ĐÁNH GIÁ Δᵢⱼ = uᵢ + vⱼ - cᵢⱼ]:")
            col_w = 10
            hdr = f"{'Δᵢⱼ':<8}|"
            for j in range(n):
                hdr += f"  B{_sub(str(j+1))}{'':>{col_w-4}}|"
            print(hdr)
            print("-" * len(hdr))
            for i in range(m):
                row_str = f"  A{_sub(str(i+1))}   |"
                for j in range(n):
                    if (i, j) in basis:
                        cell = "(cơ sở)"
                        row_str += f"  {cell:<{col_w-2}}|"
                    else:
                        val = delta_matrix[i, j]
                        cell = f"{val:g}"
                        row_str += f"  {cell:<{col_w-2}}|"
                print(row_str)
            print("-" * len(hdr))

            if entering_cell is None:
                print(f"\n=> Mọi Δᵢⱼ ≤ 0 → Không có ô vi phạm.")
                print(f"=> KẾT LUẬN: PHƯƠNG ÁN ĐÃ TỐI ƯU!")
                break

            ei, ej = entering_cell
            print(f"\n[KIỂM TRA TỐI ƯU]:")
            for i in range(m):
                for j in range(n):
                    if (i, j) not in basis and not np.isnan(delta_matrix[i, j]) and delta_matrix[i, j] > 1e-7:
                        print(f"  ! Ô ({i+1},{j+1}): Δ = {delta_matrix[i, j]:g} > 0  → VI PHẠM")

            print(f"\n  → Coi ô ({ei+1},{ej+1}) là Ô VI PHẠM lớn nhất (Δ = {max_delta:g} > 0)")
            print(f"  → Thêm ô ({ei+1},{ej+1}) vào cơ sở để tạo CHU TRÌNH điều chỉnh.")

            cycle = self._find_cycle(basis, entering_cell)
            print(f"\n[CHU TRÌNH ĐIỀU CHỈNH]:")
            cycle_parts = []
            for k, (r, c) in enumerate(cycle):
                sign = "⊕" if k % 2 == 0 else "⊖"
                val_str = f"x={X[r,c]:.0f}" if (r, c) in basis else "mới"
                cycle_parts.append(f"({r+1},{c+1})[{sign}, {val_str}]")
            print(f"  {' → '.join(cycle_parts)}")

            minus_cells = [cycle[k] for k in range(1, len(cycle), 2)]
            theta = min(X[r, c] for r, c in minus_cells)
            leaving_cell = [cell for cell in minus_cells if abs(X[cell[0], cell[1]] - theta) < 1e-7][0]

            print(f"\n  Các ô ⊖ (trừ): {', '.join([f'({r+1},{c+1})→x={X[r,c]:.0f}' for r,c in minus_cells])}")
            print(f"  → θ = min{{{', '.join([f'{X[r,c]:.0f}' for r,c in minus_cells])}}} = {theta:.0f}")

            print(f"\n[ĐIỀU CHỈNH LƯỢNG HÀNG] (θ = {theta:.0f}):")
            for k, (r, c) in enumerate(cycle):
                old_val = X[r, c]
                if k % 2 == 0:
                    new_val = old_val + theta
                    print(f"  ô ({r+1},{c+1}): {old_val:.0f} + {theta:.0f} = {new_val:.0f}  ⊕")
                else:
                    new_val = old_val - theta
                    print(f"  ô ({r+1},{c+1}): {old_val:.0f} - {theta:.0f} = {new_val:.0f}  ⊖")

            print(f"\n  → Ô ĐƯA RA khỏi cơ sở: ({leaving_cell[0]+1},{leaving_cell[1]+1}) (có θ nhỏ nhất)")

            for k, (r, c) in enumerate(cycle):
                if k % 2 == 0:
                    X[r, c] += theta
                else:
                    X[r, c] -= theta

            basis.add(entering_cell)
            basis.remove(leaving_cell)
            step += 1

        print(f"\n{'*'*70}")
        print("  KẾT QUẢ CUỐI CÙNG (BÀI TOÁN VẬN TẢI)")
        print(f"{'*'*70}")
        self._print_plan(X, basis)
        print(f"  => TỔNG CHI PHÍ VẬN CHUYỂN TỐI ƯU: F_min = {np.sum(X * self.costs):.0f}")
        print(f"{'*'*70}")
        return X

    def solve(self, init_method="lcm"):
        print("================================================================================")
        print("                         GIẢI BÀI TOÁN VẬN TẢI")
        print("================================================================================")
        self._balance_problem()
        if init_method == "nwc":
            X, basis = self.north_west_corner()
        elif init_method == "lcm":
            X, basis = self.minimum_cost_method()
        else:
            raise ValueError("Phương án ban đầu không hợp lệ. Chọn lcm hoặc nwc.")
        self.optimize_modi(X, basis)


# =====================================================================
# PHẦN 3: GIAO DIỆN ỨNG DỤNG CHUYÊN NGHIỆP (MODERN STUDIO APP)
# =====================================================================
class ModernOptimizationStudio:

    def __init__(self, root, startup_file=None):
        self.root = root
        self.current_project_path = None
        self.root.title("Optimization Studio Pro - Phần Mềm Giải Toán Tối Ưu Hóa")
        window_width = min(1600, max(800, self.root.winfo_screenwidth() - 40))
        window_height = min(920, max(600, self.root.winfo_screenheight() - 80))
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.configure(bg="#F1F5F9")

        self.font_family = tk.StringVar(value="Consolas")
        self.font_size = tk.IntVar(value=14)
        self.dark_mode_output = True

        self.colors = {
            "bg": "#F1F5F9",
            "header": "#0F172A",
            "card": "#FFFFFF",
            "primary": "#2563EB",
            "primary_hover": "#1D4ED8",
            "success": "#10B981",
            "secondary": "#64748B",
            "text": "#1E293B",
            "border": "#CBD5E1",
            "term_bg": "#0F172A",
            "term_fg": "#F8FAFC",
        }

        self._setup_styles()
        self._build_menubar()
        self._build_header()
        self._build_formatting_toolbar()
        self._build_main_workspace()

        if startup_file and os.path.exists(startup_file):
            self.load_workspace_opt(startup_file)

    # -----------------------------------------------------------------
    # 1. MENU HỆ THỐNG: LƯU / MỞ DỰ ÁN (.OPT) ĐỘC QUYỀN
    # -----------------------------------------------------------------
    def _build_menubar(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, font=("Segoe UI", 9))
        menubar.add_cascade(label="Tệp dự án (File)", menu=file_menu)
        file_menu.add_command(
            label="Create project (New)              Ctrl+N",
            command=self.new_workspace,
        )
        file_menu.add_command(
            label="Open project...   Ctrl+O",
            command=self.open_workspace_dialog,
        )
        file_menu.add_command(
            label="Save project...    Ctrl+S",
            command=self.save_workspace_opt,
        )

        file_menu.add_separator()
        file_menu.add_command(label="Thoát ứng dụng", command=self.root.quit)

        formula_menu = tk.Menu(menubar, tearoff=0, font=("Segoe UI", 9))
        menubar.add_cascade(label="Bảng Công Thức", menu=formula_menu)
        formula_menu.add_command(
            label="Công thức Bài toán Vận tải",
            command=lambda: self.show_formula_window(0),
        )
        formula_menu.add_command(
            label="Công thức Quy hoạch tuyến tính ",
            command=lambda: self.show_formula_window(1),
        )

        self.root.bind("<Control-n>", lambda e: self.new_workspace())
        self.root.bind("<Control-s>", lambda e: self.save_workspace_opt())
        self.root.bind("<Control-o>", lambda e: self.open_workspace_dialog())
        self.root.bind("<Control-b>", lambda e: self.toggle_bold_selection())
        self.root.bind("<Control-i>", lambda e: self.toggle_italic_selection())

    # -----------------------------------------------------------------
    # 2. THANH CÔNG CỤ ĐỊNH DẠNG & CÔNG THỨC TOÁN HỌC (NEW TOOLBAR)
    # -----------------------------------------------------------------
    def _build_formatting_toolbar(self):
        toolbar = tk.Frame(
            self.root,
            bg="#E2E8F0",
            height=46,
            highlightbackground="#CBD5E1",
            highlightthickness=1,
        )
        toolbar.pack(fill=tk.X, padx=16, pady=(10, 0))

        tk.Label(
            toolbar, text="Phông chữ:", font=("Segoe UI", 9, "bold"), bg="#E2E8F0", fg="#334155"
        ).pack(side=tk.LEFT, padx=(8, 4))

        font_cb = ttk.Combobox(
            toolbar,
            textvariable=self.font_family,
            values=["Consolas", "Courier New", "Segoe UI", "Times New Roman", "Arial"],
            width=14,
            state="readonly",
            font=("Segoe UI", 9),
        )
        font_cb.pack(side=tk.LEFT, padx=2, pady=6)
        font_cb.bind("<<ComboboxSelected>>", lambda e: self.apply_font_settings())

        tk.Label(
            toolbar, text="Cỡ:", font=("Segoe UI", 9, "bold"), bg="#E2E8F0", fg="#334155"
        ).pack(side=tk.LEFT, padx=(8, 2))

        size_cb = ttk.Combobox(
            toolbar,
            textvariable=self.font_size,
            values=[8, 9, 10, 11, 12, 13, 14, 16, 18, 20],
            width=4,
            state="readonly",
            font=("Segoe UI", 9),
        )
        size_cb.pack(side=tk.LEFT, padx=2, pady=6)
        size_cb.bind("<<ComboboxSelected>>", lambda e: self.apply_font_settings())

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=6)

        btn_fmt_style = {
            "bg": "#FFFFFF",
            "fg": "#1E293B",
            "relief": tk.FLAT,
            "padx": 10,
            "pady": 3,
            "cursor": "hand2",
        }

        tk.Button(
            toolbar,
            text="B",
            font=("Segoe UI", 10, "bold"),
            command=self.toggle_bold_selection,
            **btn_fmt_style,
        ).pack(side=tk.LEFT, padx=3, pady=5)

        tk.Button(
            toolbar,
            text="I",
            font=("Georgia", 10, "italic", "bold"),
            command=self.toggle_italic_selection,
            **btn_fmt_style,
        ).pack(side=tk.LEFT, padx=3, pady=5)

        tk.Button(
            toolbar,
            text="🎨 Màu chữ",
            font=("Segoe UI", 9, "bold"),
            command=self.change_selection_fg_color,
            **btn_fmt_style,
        ).pack(side=tk.LEFT, padx=3, pady=5)

        tk.Button(
            toolbar,
            text="🖍 Tô sáng",
            font=("Segoe UI", 9, "bold"),
            command=self.change_selection_bg_color,
            **btn_fmt_style,
        ).pack(side=tk.LEFT, padx=3, pady=5)

        tk.Button(
            toolbar,
            text="🌗 Nền Sáng/Tối",
            font=("Segoe UI", 9, "bold"),
            command=self.toggle_output_theme,
            **btn_fmt_style,
        ).pack(side=tk.LEFT, padx=3, pady=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=6)

        tk.Button(
            toolbar,
            text="📐 Công Thức Bài Toán",
            font=("Segoe UI", 9, "bold"),
            bg="#FEF3C7",
            fg="#92400E",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=lambda: self.show_formula_window(self.notebook.index(self.notebook.select())),
        ).pack(side=tk.LEFT, padx=4, pady=5)

        tk.Button(
            toolbar,
            text="💾 Save file",
            font=("Segoe UI", 9, "bold"),
            bg="#2563EB",
            fg="#FFFFFF",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self.save_workspace_opt,
        ).pack(side=tk.RIGHT, padx=6, pady=5)

        tk.Button(
            toolbar,
            text="📂 Open file",
            font=("Segoe UI", 9, "bold"),
            bg="#0F172A",
            fg="#38BDF8",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self.open_workspace_dialog,
        ).pack(side=tk.RIGHT, padx=4, pady=5)

    # -----------------------------------------------------------------
    # 3. CHỨC NĂNG ĐỊNH DẠNG PHÔNG CHỮ, ĐẬM, MÀU CHỮ TRÊN KẾT QUẢ
    # -----------------------------------------------------------------
    def apply_font_settings(self):
        fam = self.font_family.get()
        sz = self.font_size.get()
        self.output_text.config(font=(fam, sz))
        self.output_text.tag_configure("title", font=(fam, sz + 1, "bold"))
        self.output_text.tag_configure("step", font=(fam, sz, "bold"))
        self.output_text.tag_configure("result", font=(fam, sz, "bold"))
        self.output_text.tag_configure("error", font=(fam, sz, "bold"))
        self.output_text.tag_configure("user_bold", font=(fam, sz, "bold"))
        self.output_text.tag_configure("user_italic", font=(fam, sz, "italic"))

    def toggle_bold_selection(self):
        try:
            start, end = self.output_text.index("sel.first"), self.output_text.index("sel.last")
        except tk.TclError:
            messagebox.showinfo("Hướng dẫn", "Hãy bôi đen đoạn văn bản bên khung Lời giải trước khi bấm Tô đậm (B)!")
            return
        fam = self.font_family.get()
        sz = self.font_size.get()
        self.output_text.tag_configure("user_bold", font=(fam, sz, "bold"))
        if "user_bold" in self.output_text.tag_names("sel.first"):
            self.output_text.tag_remove("user_bold", start, end)
        else:
            self.output_text.tag_add("user_bold", start, end)

    def toggle_italic_selection(self):
        try:
            start, end = self.output_text.index("sel.first"), self.output_text.index("sel.last")
        except tk.TclError:
            messagebox.showinfo("Hướng dẫn", "Hãy bôi đen đoạn văn bản bên khung Lời giải trước khi bấm In nghiêng (I)!")
            return
        fam = self.font_family.get()
        sz = self.font_size.get()
        self.output_text.tag_configure("user_italic", font=(fam, sz, "italic"))
        if "user_italic" in self.output_text.tag_names("sel.first"):
            self.output_text.tag_remove("user_italic", start, end)
        else:
            self.output_text.tag_add("user_italic", start, end)

    def change_selection_fg_color(self):
        try:
            start, end = self.output_text.index("sel.first"), self.output_text.index("sel.last")
            has_sel = True
        except tk.TclError:
            has_sel = False

        color = colorchooser.askcolor(title="Chọn màu chữ")[1]
        if not color:
            return
        if has_sel:
            tag_name = f"fg_{color.replace('#', '')}"
            self.output_text.tag_configure(tag_name, foreground=color)
            self.output_text.tag_add(tag_name, start, end)
        else:
            self.output_text.config(fg=color)

    def change_selection_bg_color(self):
        try:
            start, end = self.output_text.index("sel.first"), self.output_text.index("sel.last")
        except tk.TclError:
            messagebox.showinfo("Hướng dẫn", "Hãy bôi đen đoạn văn bản bên khung Lời giải cần Tô sáng (Highlight)!")
            return
        color = colorchooser.askcolor(title="Chọn màu nền tô sáng")[1]
        if color:
            tag_name = f"bg_{color.replace('#', '')}"
            self.output_text.tag_configure(tag_name, background=color, foreground="#000000")
            self.output_text.tag_add(tag_name, start, end)

    def toggle_output_theme(self):
        self.dark_mode_output = not self.dark_mode_output
        if self.dark_mode_output:
            self.output_text.config(bg="#0F172A", fg="#F8FAFC", insertbackground="white")
            self.output_text.tag_configure("title", foreground="#38BDF8")
            self.output_text.tag_configure("step", foreground="#FBBF24")
            self.output_text.tag_configure("result", foreground="#34D399")
            self.output_text.tag_configure("arrow", foreground="#A78BFA")
        else:
            self.output_text.config(bg="#FFFFFF", fg="#0F172A", insertbackground="black")
            self.output_text.tag_configure("title", foreground="#1D4ED8")
            self.output_text.tag_configure("step", foreground="#B45309")
            self.output_text.tag_configure("result", foreground="#047857")
            self.output_text.tag_configure("arrow", foreground="#6D28D9")

    # -----------------------------------------------------------------
    # 4. CỬA SỔ TRA CỨU CÔNG THỨC TỪNG BÀI TOÁN
    # -----------------------------------------------------------------
    def show_formula_window(self, default_tab=0):
        win = tk.Toplevel(self.root)
        win.title("Sổ Tay Công Thức & Phương Pháp Giải Tối Ưu Hóa")
        win.geometry("820x600")
        win.configure(bg="#F8FAFC")

        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        formulas = [
            (
                " 1. Bài Toán Vận Tải ",
                """====================================================================
BÀI TOÁN VẬN TẢI (TRANSPORTATION PROBLEM)
====================================================================

1. MÔ HÌNH TOÁN HỌC:
   • Hàm mục tiêu (Cực tiểu tổng chi phí vận chuyển):
       F(x) = ∑ᵢ₌₁..ₘ ∑ⱼ₌₁..ₙ  cᵢⱼ · xᵢⱼ  -->  MIN
   • Ràng buộc trạm phát (m trạm):
       ∑ⱼ₌₁..ₙ xᵢⱼ = aᵢ   (i = 1..m)
   • Ràng buộc trạm thu (n trạm):
       ∑ᵢ₌₁..ₘ xᵢⱼ = bⱼ   (j = 1..n)
   • Điều kiện không âm: xᵢⱼ ≥ 0

2. ĐIỀU KIỆN CÂN BẰNG THU - PHÁT:
   • Bài toán cân bằng khi:  ∑aᵢ = ∑bⱼ
   • Nếu ∑aᵢ > ∑bⱼ (Phát > Thu): Thêm 1 trạm thu giả Bₙ₊₁ với bₙ₊₁ = ∑aᵢ - ∑bⱼ, cước phí cᵢ,ₙ₊₁ = 0.
   • Nếu ∑aᵢ < ∑bⱼ (Thu > Phát): Thêm 1 trạm phát giả Aₘ₊₁ với aₘ₊₁ = ∑bⱼ - ∑aᵢ, cước phí cₘ₊₁,ⱼ = 0.

3. TÌM PHƯƠNG ÁN CỰC BIÊN BAN ĐẦU (m + n - 1 ô chọn):
   • Góc Tây Bắc (NWC): Phân phối từ ô (1,1) tối đa xᵢⱼ = min(aᵢ, bⱼ), gạch dòng/cột đã thỏa mãn.
   • Cước phí nhỏ nhất (LCM): Ưu tiên phân phối tối đa vào ô có cước phí cᵢⱼ nhỏ nhất trên bảng.

4. THUẬT TOÁN THẾ VỊ KIỂM TRA & TỐI ƯU:
   • Bước 1: Tính các thế vị hàng uᵢ và cột vⱼ dựa trên các Ô CƠ SỞ:
       uᵢ + vⱼ = cᵢⱼ   (Chọn cố định u₁ = 0)
   • Bước 2: Tính hệ số ước lượng cho các Ô LOẠI:
       Δᵢⱼ = uᵢ + vⱼ - cᵢⱼ
   • Bước 3 (Tiêu chuẩn tối ưu):
       - Nếu mọi Δᵢⱼ ≤ 0: Phương án hiện tại ĐÃ TỐI ƯU.
       - Nếu tồn tại Δᵢⱼ > 0: Chọn ô (r, s) có Δᵣₛ > 0 lớn nhất đưa vào cơ sở.
   • Bước 4: Lập chu trình đóng (±) xuất phát từ ô (r, s) đi qua các ô cơ sở.
       - Lượng hàng điều chỉnh: θ = min { xᵢⱼ tại các đỉnh mang dấu âm (-) }.
       - Cộng θ vào các ô mang dấu (+), trừ θ ở các ô mang dấu (-).""",
            ),
            (
                " 2. Quy Hoạch Tuyến Tính ",
                """====================================================================
BÀI TOÁN QUY HOẠCH TUYẾN TÍNH (LINEAR PROGRAMMING)
====================================================================

1. DẠNG TỔNG QUÁT & ĐƯA VỀ DẠNG CHUẨN:
   • Hàm mục tiêu:  Z = c₁x₁ + c₂x₂ + ... + cₙxₙ  -->  MAX (hoặc MIN)
     (Nếu bài toán tìm MIN, ta đổi dấu hàm mục tiêu: Z' = -Z --> MAX)
   • Chuẩn hóa vế phải bᵢ ≥ 0 (Nếu bᵢ < 0 thì nhân cả 2 vế ràng buộc với -1 và đảo chiều bất đẳng thức).

2. THÊM BIẾN PHỤ & BIẾN GIẢ:
   • Ràng buộc dạng (≤):  ∑aᵢⱼxⱼ + xₖ = bᵢ          (Hệ số mục tiêu của xₖ là 0)
   • Ràng buộc dạng (≥):  ∑aᵢⱼxⱼ - xₖ + aᵢ = bᵢ     (Hệ số mục tiêu của aᵢ là -M)
     xₖ là biến phụ không âm, đánh số tiếp sau các biến ban đầu: xₙ₊₁, xₙ₊₂, ...
   • Ràng buộc dạng (=):  ∑aᵢⱼxⱼ + aᵢ = bᵢ          (Hệ số mục tiêu của aᵢ là -M, với M >> 0 rất lớn)

3. CÔNG THỨC TÍNH TRÊN BẢNG ĐƠN HÌNH:
   • Giá trị hàm mục tiêu tại phương án cơ sở Xᴮ:
       Z = ∑ Cᴮᵢ · bᵢ
   • Hệ số ước lượng tại cột j:
       Δⱼ = (∑ Cᴮᵢ · aᵢⱼ) - cⱼ
   • Điều kiện tối ưu (Bài toán MAX):
       - Nếu mọi Δⱼ ≥ 0: Phương án hiện tại ĐÃ TỐI ƯU.
         (Chú ý: Nếu trong cơ sở tối ưu vẫn còn biến giả aᵢ > 0 thì bài toán VÔ NGHIỆM).
       - Nếu tồn tại Δⱼ < 0: Chọn cột k có Δₖ < 0 nhỏ nhất (âm sâu nhất) làm BIẾN VÀO.
         + Nếu mọi aᵢₖ ≤ 0 trên cột k: Bài toán KHÔNG GIỚI NỘI (Unbounded).
         + Ngược lại, tính tỷ số:  θ = min { bᵢ / aᵢₖ  với  aᵢₖ > 0 } = bᵣ / aᵣₖ.
         + Hàng r xác định BIẾN RA, phần tử aᵣₖ là PHẦN TỬ XOAY (Pivot).

4. CÔNG THỨC BIẾN ĐỔI BẢNG (QUY TẮC HÌNH CHỮ NHẬT / GAUSS-JORDAN):
   • Dòng xoay mới = (Dòng xoay cũ) / aᵣₖ
   • Các dòng khác mới = (Dòng cũ) - (Phần tử trên cột xoay) × (Dòng xoay mới)""",
            ),
        ]

        for title, content in formulas:
            frame = tk.Frame(nb, bg="#FFFFFF")
            nb.add(frame, text=title)
            txt = tk.Text(
                frame,
                font=("Consolas", 11),
                bg="#FFFFFF",
                fg="#1E293B",
                padx=20,
                pady=16,
                spacing1=3,
                spacing3=3,
                wrap=tk.WORD,
                relief=tk.FLAT,
            )
            txt.insert("1.0", content)
            txt.tag_configure(
                "formula_title", foreground="#1E40AF", background="#EFF6FF",
                font=("Segoe UI", 14, "bold"), spacing1=12, spacing3=12,
                lmargin1=10, lmargin2=10,
            )
            txt.tag_configure(
                "formula_heading", foreground="#0F766E", background="#F0FDFA",
                font=("Segoe UI", 11, "bold"), spacing1=14, spacing3=8,
                lmargin1=8, lmargin2=8,
            )
            txt.tag_configure(
                "formula_separator", foreground="#CBD5E1", spacing1=0, spacing3=0,
            )
            txt.tag_configure("formula_bullet", foreground="#334155")
            for line_number, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                prefix, dot, _ = stripped.partition(".")
                tag = None
                if stripped and set(stripped) == {"="}:
                    tag = "formula_separator"
                elif line_number == 2:
                    tag = "formula_title"
                elif dot and prefix.isdigit():
                    tag = "formula_heading"
                elif stripped.startswith("•"):
                    tag = "formula_bullet"
                if tag:
                    txt.tag_add(tag, f"{line_number}.0", f"{line_number}.end")
            txt.config(state=tk.DISABLED)
            scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=txt.yview)
            txt.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        if 0 <= default_tab < 2:
            nb.select(default_tab)

    # -----------------------------------------------------------------
    # 5. LƯU & MỞ DỰ ÁN (.OPT) ĐỘC QUYỀN (CẢ ĐỀ BÀI + LỜI GIẢI + TAGS)
    # -----------------------------------------------------------------
    def _collect_workspace_state(self):
        saved_tags = {}
        tag_configs = {}
        for tag in self.output_text.tag_names():
            if tag in ("sel",):
                continue
            ranges = [str(r) for r in self.output_text.tag_ranges(tag)]
            if ranges:
                saved_tags[tag] = ranges
                fg = self.output_text.tag_cget(tag, "foreground")
                bg = self.output_text.tag_cget(tag, "background")
                fnt = self.output_text.tag_cget(tag, "font")
                tag_configs[tag] = {"fg": fg, "bg": bg, "font": fnt}

        return {
            "app_signature": "OPTIMIZATION_STUDIO_PRO_V2",
            "active_tab": self.notebook.index(self.notebook.select()),
            "font_family": self.font_family.get(),
            "font_size": self.font_size.get(),
            "dark_mode_output": self.dark_mode_output,
            "tp": {
                "m": self.tp_m_var.get(),
                "n": self.tp_n_var.get(),
                "method": self.tp_method_cb.get(),
                "costs": [[e.get() for e in row] for row in self.tp_cost_entries],
                "supply": [e.get() for e in self.tp_supply_entries],
                "demand": [e.get() for e in self.tp_demand_entries],
            },
            "lp": {
                "n": self.lp_n_var.get(),
                "m": self.lp_m_var.get(),
                "mode": self.lp_mode_cb.get(),
                "c": [e.get() for e in self.lp_c_entries],
                "A": [[e.get() for e in row] for row in self.lp_A_entries],
                "signs": [cb.get() for cb in self.lp_sign_cbs],
                "b": [e.get() for e in self.lp_b_entries],
            },
            "solution_text": self.output_text.get("1.0", "end-1c"),
            "solution_tags": saved_tags,
            "tag_configs": tag_configs,
        }

    def save_workspace_opt(self):
        state = self._collect_workspace_state()
        filepath = filedialog.asksaveasfilename(
            defaultextension=".opt",
            filetypes=[("Optimization", "*.opt")],
            initialfile="Tối Ưu Hoá",
            title="Save as",
        )
        if not filepath:
            return
        try:
            raw_json = json.dumps(state, ensure_ascii=False).encode("utf-8")
            compressed = zlib.compress(raw_json, level=9)
            encoded = b"OPTPRO2!" + base64.b85encode(compressed)
            with open(filepath, "wb") as f:
                f.write(encoded)
            self.current_project_path = filepath
            self.root.title(f"Optimization Studio Pro - [{os.path.basename(filepath)}]")
            messagebox.showinfo(
                "Đã lưu dự án",
                f"Đã lưu cả đề bài và lời giải vào file dự án:\n{filepath}\n\n"
                "Khi mở file .opt này, toàn bộ giao diện đề bài và các bước giải sẽ hiện lại nguyên vẹn!",
            )
        except Exception as e:
            messagebox.showerror("Lỗi lưu dự án", f"Không thể lưu file: {e}")

    def open_workspace_dialog(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Optimization Studio Project (*.opt)", "*.opt")],
            title="Mở file dự án (.opt)",
        )
        if filepath:
            self.load_workspace_opt(filepath)

    def load_workspace_opt(self, filepath):
        try:
            with open(filepath, "rb") as f:
                raw_bytes = f.read()
            if not raw_bytes.startswith(b"OPTPRO2!"):
                raise ValueError("File không đúng định dạng dự án Optimization Studio (.opt)!")

            compressed = base64.b85decode(raw_bytes[8:])
            state = json.loads(zlib.decompress(compressed).decode("utf-8"))

            # 1. Khôi phục Tab Vận tải
            if "tp" in state:
                tp = state["tp"]
                self.tp_m_var.set(tp["m"])
                self.tp_n_var.set(tp["n"])
                self._rebuild_tp_grid()
                methods = self.tp_method_cb["values"]
                saved_method = tp.get("method")
                self.tp_method_cb.set(saved_method if saved_method in methods else methods[0])
                for i in range(tp["m"]):
                    for j in range(tp["n"]):
                        self.tp_cost_entries[i][j].delete(0, tk.END)
                        self.tp_cost_entries[i][j].insert(0, tp["costs"][i][j])
                    self.tp_supply_entries[i].delete(0, tk.END)
                    self.tp_supply_entries[i].insert(0, tp["supply"][i])
                for j, val in enumerate(tp["demand"]):
                    self.tp_demand_entries[j].delete(0, tk.END)
                    self.tp_demand_entries[j].insert(0, val)

            # 2. Khôi phục Tab QHTT
            if "lp" in state:
                lp = state["lp"]
                self.lp_n_var.set(lp["n"])
                self.lp_m_var.set(lp["m"])
                self._rebuild_lp_grid()
                self.lp_mode_cb.set(lp["mode"])
                for j, val in enumerate(lp["c"]):
                    self.lp_c_entries[j].delete(0, tk.END)
                    self.lp_c_entries[j].insert(0, val)
                for i in range(lp["m"]):
                    for j in range(lp["n"]):
                        self.lp_A_entries[i][j].delete(0, tk.END)
                        self.lp_A_entries[i][j].insert(0, lp["A"][i][j])
                    self.lp_sign_cbs[i].set(lp["signs"][i])
                    self.lp_b_entries[i].delete(0, tk.END)
                    self.lp_b_entries[i].insert(0, lp["b"][i])

            # 3. Khôi phục phông chữ, giao diện & toàn bộ lời giải + màu sắc
            self.font_family.set(state.get("font_family", "Consolas"))
            self.font_size.set(max(14, state.get("font_size", 14)))
            self.dark_mode_output = not state.get("dark_mode_output", True)
            self.toggle_output_theme()
            self.apply_font_settings()

            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", state.get("solution_text", ""))

            tag_configs = state.get("tag_configs", {})
            for tag, cfg in tag_configs.items():
                kwargs = {}
                if cfg.get("fg"):
                    kwargs["foreground"] = cfg["fg"]
                if cfg.get("bg"):
                    kwargs["background"] = cfg["bg"]
                if cfg.get("font"):
                    kwargs["font"] = cfg["font"]
                if kwargs:
                    self.output_text.tag_configure(tag, **kwargs)

            for tag, ranges in state.get("solution_tags", {}).items():
                for k in range(0, len(ranges), 2):
                    self.output_text.tag_add(tag, ranges[k], ranges[k + 1])

            active_idx = state.get("active_tab", 0)
            if 0 <= active_idx < 2:
                self.notebook.select(active_idx)
            self.current_project_path = filepath
            self.root.title(f"Optimization Studio Pro - [{os.path.basename(filepath)}]")
        except Exception as e:
            messagebox.showerror("Lỗi mở dự án", f"Không thể đọc file dự án .opt:\n{e}")

    def new_workspace(self):
        idx = self.notebook.index(self.notebook.select())
        if idx == 0:
            self._rebuild_tp_grid()
        elif idx == 1:
            self._rebuild_lp_grid()
        self.output_text.delete("1.0", tk.END)
        self.current_project_path = None
        self.root.title("Optimization Studio Pro - Phần Mềm Giải Toán Tối Ưu Hóa")

    def register_file_association_windows(self):
        if sys.platform != "win32":
            messagebox.showinfo("Thông báo", "Tính năng liên kết file trực tiếp chỉ hỗ trợ trên Windows.")
            return
        try:
            import winreg

            if getattr(sys, "frozen", False):
                cmd = f'"{sys.executable}" "%1"'
            else:
                script_path = os.path.abspath(sys.argv[0])
                cmd = f'"{sys.executable}" "{script_path}" "%1"'

            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\.opt") as k:
                winreg.SetValue(k, "", winreg.REG_SZ, "OptimizationStudio.Project")

            with winreg.CreateKey(
                winreg.HKEY_CURRENT_USER, r"Software\Classes\OptimizationStudio.Project"
            ) as k:
                winreg.SetValue(k, "", winreg.REG_SZ, "Optimization Studio Project File")

            with winreg.CreateKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Classes\OptimizationStudio.Project\shell\open\command",
            ) as k:
                winreg.SetValue(k, "", winreg.REG_SZ, cmd)

            messagebox.showinfo(
                "Đăng ký thành công",
                "Đã liên kết đuôi file .opt với phần mềm này!\n"
                "Từ giờ khi bạn nhấp đúp vào file .opt, Windows sẽ mở thẳng lên giao diện ứng dụng.",
            )
        except Exception as e:
            messagebox.showerror("Lỗi đăng ký", f"Không thể đăng ký đuôi file: {e}")

    # -----------------------------------------------------------------
    # 6. CẤU HÌNH GIAO DIỆN CHÍNH & 2 TAB BÀI TOÁN
    # -----------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 9, "bold"),
            padding=[12, 6],
            background="#E2E8F0",
            foreground="#475569",
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.colors["primary"])],
            foreground=[("selected", "#FFFFFF")],
            font=[("selected", ("Segoe UI", 10, "bold"))],
            padding=[("selected", [18, 10])],
            expand=[("selected", [1, 4, 1, 0])],  
        )
        style.configure("Card.TFrame", background=self.colors["card"])

    def _create_btn(self, parent, text, bg_color, command, width=None):
        btn = tk.Button(
            parent,
            text=text,
            font=("Segoe UI", 10, "bold"),
            bg=bg_color,
            fg="white",
            activebackground=self.colors["primary_hover"],
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            padx=14,
            pady=7,
            command=command,
        )
        if width:
            btn.config(width=width)
        return btn

    def _build_header(self):
        header = tk.Frame(self.root, bg=self.colors["header"], height=64)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title_box = tk.Frame(header, bg=self.colors["header"])
        title_box.pack(side=tk.LEFT, padx=24, pady=8)

        tk.Label(
            title_box,
            text="OPTIMIZATION ",
            font=("Segoe UI", 15, "bold"),
            bg=self.colors["header"],
            fg="#38BDF8",
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="by Nhóm 6 - 20261IT6095001 ",
            font=("Segoe UI", 9, "italic"),
            bg=self.colors["header"],
            fg="#94A3B8",
        ).pack(anchor="w")

        btn_clear = self._create_btn(
            header,
            "Xóa Màn Hình Kết Quả",
            "#334155",
            lambda: self.output_text.delete("1.0", tk.END),
        )
        btn_clear.pack(side=tk.RIGHT, padx=24, pady=12)

    def _build_main_workspace(self):
        main_pane = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bg=self.colors["bg"],
            sashwidth=6,
            sashrelief=tk.FLAT,
        )
        main_pane.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        left_container = tk.Frame(main_pane, bg=self.colors["bg"])
        main_pane.add(left_container, width=580, minsize=420, stretch="never")

        self.notebook = ttk.Notebook(left_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        right_container = tk.Frame(
            main_pane,
            bg=self.colors["term_bg"],
            highlightbackground="#334155",
            highlightthickness=1,
        )
        main_pane.add(right_container, width=980, minsize=360, stretch="always")

        term_header = tk.Frame(right_container, bg="#1E293B", height=36)
        term_header.pack(fill=tk.X)
        term_header.pack_propagate(False)

        tk.Label(
            term_header,
            text="  BẢNG TRÌNH BÀY CHI TIẾT TỪNG BƯỚC GIẢI ",
            font=("Segoe UI", 9, "bold"),
            bg="#1E293B",
            fg="#E2E8F0",
        ).pack(side=tk.LEFT, padx=8)

        text_frame = tk.Frame(right_container, bg=self.colors["term_bg"])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        scrollbar_y = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        scrollbar_x = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL)

        self.output_text = tk.Text(
            text_frame,
            font=(self.font_family.get(), self.font_size.get()),
            bg=self.colors["term_bg"],
            fg=self.colors["term_fg"],
            insertbackground="white",
            relief=tk.FLAT,
            wrap=tk.NONE,
            padx=12,
            pady=10,
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set,
        )
        scrollbar_y.config(command=self.output_text.yview)
        scrollbar_x.config(command=self.output_text.xview)

        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.output_text.tag_configure("title", foreground="#38BDF8", font=(self.font_family.get(), 15, "bold"))
        self.output_text.tag_configure("step", foreground="#FBBF24", font=(self.font_family.get(), 14, "bold"))
        self.output_text.tag_configure("result", foreground="#34D399", font=(self.font_family.get(), 14, "bold"))
        self.output_text.tag_configure("arrow", foreground="#A78BFA")
        self.output_text.tag_configure("error", foreground="#F87171", font=(self.font_family.get(), 14, "bold"))

        self._init_tab_transport()
        self._init_tab_lp()

    def _render_colored_output(self, content):
        self.output_text.delete("1.0", tk.END)
        for line in content.split("\n"):
            tag = None
            if line.startswith("===") or "GIẢI BÀI TOÁN" in line:
                tag = "title"
            elif line.startswith("--- BƯỚC") or line.startswith("["):
                tag = "step"
            elif (
                line.startswith("***")
                or "KẾT QUẢ" in line
                or "KẾT LUẬN" in line
                or "TỐI ƯU" in line
                or "=> " in line
            ):
                tag = "result"
            elif "->" in line or "→" in line or "+" in line:
                tag = "arrow"
            elif "VÔ NGHIỆM" in line or "KHÔNG GIỚI NỘI" in line or "VI PHẠM" in line:
                tag = "error"

            if tag:
                self.output_text.insert(tk.END, line + "\n", tag)
            else:
                self.output_text.insert(tk.END, line + "\n")

    def _execute_solver(self, task_func):
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        try:
            task_func()
        except Exception as e:
            messagebox.showerror("Lỗi nhập liệu", f"Vui lòng kiểm tra lại các ô số liệu!\nChi tiết: {e}")
        finally:
            sys.stdout = old_stdout
        self._render_colored_output(buffer.getvalue())

    # =================================================================
    # TAB 1: BÀI TOÁN VẬN TẢI
    # =================================================================
    def _init_tab_transport(self):
        tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.notebook.add(tab, text=" 1. Bài Toán Vận Tải ")

        cfg_bar = tk.Frame(tab, bg="#F8FAFC", padx=12, pady=10)
        cfg_bar.pack(fill=tk.X, padx=12, pady=10)

        tk.Label(cfg_bar, text="Trạm phát (m):", font=("Segoe UI", 9, "bold"), bg="#F8FAFC").pack(side=tk.LEFT)
        self.tp_m_var = tk.IntVar(value=3)
        tk.Spinbox(cfg_bar, from_=2, to=6, textvariable=self.tp_m_var, width=3, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=4)

        tk.Label(cfg_bar, text="Trạm thu (n):", font=("Segoe UI", 9, "bold"), bg="#F8FAFC").pack(side=tk.LEFT, padx=(6, 0))
        self.tp_n_var = tk.IntVar(value=4)
        tk.Spinbox(cfg_bar, from_=2, to=6, textvariable=self.tp_n_var, width=3, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=4)

        self._create_btn(cfg_bar, "Tạo Bảng", self.colors["secondary"], self._rebuild_tp_grid).pack(side=tk.LEFT, padx=8)

        tk.Label(cfg_bar, text="Phương Án :", font=("Segoe UI", 9, "bold"), bg="#F8FAFC").pack(side=tk.LEFT, padx=(6, 2))
        self.tp_method_cb = ttk.Combobox(
            cfg_bar,
            values=["Cước phí nhỏ nhất (LCM)", "Góc Tây Bắc (NWC)"],
            width=21,
            state="readonly",
            font=("Segoe UI", 9),
        )
        self.tp_method_cb.current(0)
        self.tp_method_cb.pack(side=tk.LEFT, padx=4)

        self.tp_grid_container = tk.Frame(tab, bg=self.colors["card"])
        self.tp_grid_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        action_bar = tk.Frame(tab, bg=self.colors["card"], pady=12)
        action_bar.pack(fill=tk.X, padx=15, side=tk.BOTTOM)

        self._create_btn(action_bar, "Nạp Đề Mẫu", self.colors["success"], self._load_tp_sample_grid).pack(side=tk.LEFT, padx=5)
        self._create_btn(action_bar, "GIẢI BÀI TOÁN VẬN TẢI ▶", self.colors["primary"], self._solve_tp_from_grid).pack(side=tk.RIGHT, padx=5)

        self._load_tp_sample_grid()

    def _rebuild_tp_grid(self):
        for widget in self.tp_grid_container.winfo_children():
            widget.destroy()

        m = self.tp_m_var.get()
        n = self.tp_n_var.get()

        box = tk.LabelFrame(
            self.tp_grid_container,
            text=" Bảng Ma Trận Cước Phí Cᵢⱼ, Lượng Phát (aᵢ) & Lượng Thu (bⱼ) ",
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg=self.colors["primary"],
            padx=10,
            pady=10,
        )
        box.pack(fill=tk.BOTH, expand=True)

        tk.Label(box, text="Phát \\ Thu", font=("Segoe UI", 9, "bold"), bg="white").grid(row=0, column=0, padx=6, pady=6)
        for j in range(n):
            tk.Label(box, text=f"Trạm B{_sub(str(j+1))}", font=("Segoe UI", 9, "bold"), bg="white", fg="#1E293B").grid(row=0, column=j + 1, padx=6, pady=6)
        tk.Label(box, text="Lượng Phát (aᵢ)", font=("Segoe UI", 9, "bold"), bg="white", fg="#2563EB").grid(row=0, column=n + 1, padx=8, pady=6)

        self.tp_cost_entries = []
        self.tp_supply_entries = []

        for i in range(m):
            tk.Label(box, text=f"Trạm A{_sub(str(i+1))}:", font=("Segoe UI", 10, "bold"), bg="white").grid(row=i + 1, column=0, padx=6, pady=6, sticky="w")
            row_ents = []
            for j in range(n):
                e = tk.Entry(box, width=7, font=("Segoe UI", 10), justify="center", bg="#F8FAFC", relief=tk.SOLID, bd=1)
                e.insert(0, "0")
                e.grid(row=i + 1, column=j + 1, padx=5, pady=6)
                row_ents.append(e)
            self.tp_cost_entries.append(row_ents)

            s_e = tk.Entry(box, width=9, font=("Segoe UI", 10, "bold"), justify="center", bg="#EFF6FF", relief=tk.SOLID, bd=1)
            s_e.insert(0, "0")
            s_e.grid(row=i + 1, column=n + 1, padx=8, pady=6)
            self.tp_supply_entries.append(s_e)

        tk.Label(box, text="Lượng Thu (bⱼ):", font=("Segoe UI", 9, "bold"), bg="white", fg="#059669").grid(row=m + 1, column=0, padx=6, pady=12, sticky="w")
        self.tp_demand_entries = []
        for j in range(n):
            d_e = tk.Entry(box, width=7, font=("Segoe UI", 10, "bold"), justify="center", bg="#ECFDF5", relief=tk.SOLID, bd=1)
            d_e.insert(0, "0")
            d_e.grid(row=m + 1, column=j + 1, padx=5, pady=12)
            self.tp_demand_entries.append(d_e)

    def _load_tp_sample_grid(self):
        self.tp_m_var.set(3)
        self.tp_n_var.set(4)
        self._rebuild_tp_grid()
        costs_sample = [[4, 3, 6, 2], [7, 5, 2, 3], [2, 6, 4, 5]]
        supply_sample = [30, 40, 50]
        demand_sample = [20, 35, 25, 40]
        for i in range(3):
            for j in range(4):
                self.tp_cost_entries[i][j].delete(0, tk.END)
                self.tp_cost_entries[i][j].insert(0, str(costs_sample[i][j]))
            self.tp_supply_entries[i].delete(0, tk.END)
            self.tp_supply_entries[i].insert(0, str(supply_sample[i]))
        for j in range(4):
            self.tp_demand_entries[j].delete(0, tk.END)
            self.tp_demand_entries[j].insert(0, str(demand_sample[j]))

    def _solve_tp_from_grid(self):
        def task():
            costs = [[float(e.get()) for e in row] for row in self.tp_cost_entries]
            supply = [float(e.get()) for e in self.tp_supply_entries]
            demand = [float(e.get()) for e in self.tp_demand_entries]
            sel = self.tp_method_cb.get()
            method = "nwc" if "NWC" in sel else "lcm"
            solver = TransportationSolver(supply, demand, costs)
            solver.solve(init_method=method)

        self._execute_solver(task)

    # =================================================================
    # TAB 2: QUY HOẠCH TUYẾN TÍNH
    # =================================================================
    def _init_tab_lp(self):
        tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.notebook.add(tab, text=" 2. Quy Hoạch Tuyến Tính ")

        cfg_bar = tk.Frame(tab, bg="#F8FAFC", padx=15, pady=12)
        cfg_bar.pack(fill=tk.X, padx=12, pady=12)

        tk.Label(cfg_bar, text="Số biến (n):", font=("Segoe UI", 10, "bold"), bg="#F8FAFC").pack(side=tk.LEFT)
        self.lp_n_var = tk.IntVar(value=2)
        tk.Spinbox(cfg_bar, from_=2, to=8, textvariable=self.lp_n_var, width=4, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=6)

        tk.Label(cfg_bar, text="Số ràng buộc (m):", font=("Segoe UI", 10, "bold"), bg="#F8FAFC").pack(side=tk.LEFT, padx=(10, 0))
        self.lp_m_var = tk.IntVar(value=3)
        tk.Spinbox(cfg_bar, from_=1, to=8, textvariable=self.lp_m_var, width=4, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=6)

        self._create_btn(cfg_bar, "Tạo Bảng Nhập", self.colors["secondary"], self._rebuild_lp_grid).pack(side=tk.LEFT, padx=12)

        self.lp_grid_container = tk.Frame(tab, bg=self.colors["card"])
        self.lp_grid_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        action_bar = tk.Frame(tab, bg=self.colors["card"], pady=12)
        action_bar.pack(fill=tk.X, padx=15, side=tk.BOTTOM)

        self._create_btn(action_bar, "Nạp Đề Mẫu", self.colors["success"], self._load_lp_sample_grid).pack(side=tk.LEFT, padx=5)
        self._create_btn(action_bar, "GIẢI BÀI TOÁN QHTT ▶", self.colors["primary"], self._solve_lp_from_grid).pack(side=tk.RIGHT, padx=5)
        self._create_btn(action_bar, "LẬP ĐỐI NGẪU", self.colors["secondary"], self._show_dual_from_grid).pack(side=tk.RIGHT, padx=5)

        self._load_lp_sample_grid()

    def _rebuild_lp_grid(self):
        for widget in self.lp_grid_container.winfo_children():
            widget.destroy()

        n = self.lp_n_var.get()
        m = self.lp_m_var.get()

        obj_frame = tk.LabelFrame(
            self.lp_grid_container,
            text=" Hàm Mục Tiêu (Z) ",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["card"],
            fg=self.colors["primary"],
            padx=10,
            pady=10,
        )
        obj_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Label(obj_frame, text="Z =", font=("Segoe UI", 10, "bold"), bg="white").pack(side=tk.LEFT, padx=4)
        self.lp_c_entries = []
        for j in range(n):
            ent = tk.Entry(obj_frame, width=6, font=("Segoe UI", 10), justify="center", bg="#F8FAFC", relief=tk.SOLID, bd=1)
            ent.insert(0, "0")
            ent.pack(side=tk.LEFT, padx=2)
            self.lp_c_entries.append(ent)
            lbl_txt = f"x{_sub(str(j+1))}  +" if j < n - 1 else f"x{_sub(str(j+1))}   →"
            tk.Label(obj_frame, text=lbl_txt, font=("Segoe UI", 10), bg="white").pack(side=tk.LEFT, padx=2)

        self.lp_mode_cb = ttk.Combobox(obj_frame, values=["MAX", "MIN"], width=6, state="readonly", font=("Segoe UI", 10, "bold"))
        self.lp_mode_cb.set("MAX")
        self.lp_mode_cb.pack(side=tk.LEFT, padx=6)

        con_frame = tk.LabelFrame(
            self.lp_grid_container,
            text=" Hệ Ràng Buộc Tuyến Tính ",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["card"],
            fg=self.colors["primary"],
            padx=10,
            pady=10,
        )
        con_frame.pack(fill=tk.BOTH, expand=True)

        self.lp_A_entries = []
        self.lp_sign_cbs = []
        self.lp_b_entries = []

        for i in range(m):
            row_entries = []
            tk.Label(con_frame, text=f"R.Buộc {i+1}:", font=("Segoe UI", 9, "bold"), bg="white", fg="#64748B").grid(row=i, column=0, padx=4, pady=6, sticky="w")
            for j in range(n):
                ent = tk.Entry(con_frame, width=6, font=("Segoe UI", 10), justify="center", bg="#F8FAFC", relief=tk.SOLID, bd=1)
                ent.insert(0, "0")
                ent.grid(row=i, column=2 * j + 1, padx=2, pady=6)
                row_entries.append(ent)
                var_lbl = f"x{_sub(str(j+1))} +" if j < n - 1 else f"x{_sub(str(j+1))}"
                tk.Label(con_frame, text=var_lbl, font=("Segoe UI", 10), bg="white").grid(row=i, column=2 * j + 2, padx=2)

            sign_cb = ttk.Combobox(con_frame, values=["≤", "≥", "="], width=3, state="readonly", font=("Segoe UI", 10, "bold"))
            sign_cb.set("≤")
            sign_cb.grid(row=i, column=2 * n + 1, padx=6, pady=6)

            b_ent = tk.Entry(con_frame, width=8, font=("Segoe UI", 10, "bold"), justify="center", bg="#EFF6FF", relief=tk.SOLID, bd=1)
            b_ent.insert(0, "0")
            b_ent.grid(row=i, column=2 * n + 2, padx=4, pady=6)

            self.lp_A_entries.append(row_entries)
            self.lp_sign_cbs.append(sign_cb)
            self.lp_b_entries.append(b_ent)

    def _load_lp_sample_grid(self):
        self.lp_n_var.set(2)
        self.lp_m_var.set(3)
        self._rebuild_lp_grid()
        c_sample = [3, 2]
        A_sample = [[2, 1], [2, 3], [3, 1]]
        signs_sample = ["≤", "≤", "≥"]
        b_sample = [18, 42, 6]
        for j in range(2):
            self.lp_c_entries[j].delete(0, tk.END)
            self.lp_c_entries[j].insert(0, str(c_sample[j]))
        for i in range(3):
            for j in range(2):
                self.lp_A_entries[i][j].delete(0, tk.END)
                self.lp_A_entries[i][j].insert(0, str(A_sample[i][j]))
            self.lp_sign_cbs[i].set(signs_sample[i])
            self.lp_b_entries[i].delete(0, tk.END)
            self.lp_b_entries[i].insert(0, str(b_sample[i]))

    def _show_dual_from_grid(self):
        self._solve_lp_from_grid(dual_only=True)

    def _solve_lp_from_grid(self, dual_only=False):
        def task():
            c = [float(e.get()) for e in self.lp_c_entries]
            A = [[float(e.get()) for e in row] for row in self.lp_A_entries]
            signs = [cb.get() for cb in self.lp_sign_cbs]
            b = [float(e.get()) for e in self.lp_b_entries]
            mode = self.lp_mode_cb.get().lower()
            solver = LinearProgrammingSolver(c, A, b, signs, mode)
            if dual_only:
                solver.print_dual()
            else:
                solver.solve()

        self._execute_solver(task)


if __name__ == "__main__":
    startup_opt_file = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    app = ModernOptimizationStudio(root, startup_file=startup_opt_file)
    root.mainloop()
